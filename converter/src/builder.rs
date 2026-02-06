// src/builder.rs
// ============================================================================
// BUILDER - Orquestador genérico de conversión
// ============================================================================
//
// El builder es TONTO:
// - NO decide bloques (lo hace el CLI)
// - NO decide cuantización (lo sugiere el mapper)
// - Solo lee, cuantiza, escribe
//
// v9.0.5: Prefijo text. para consistencia (todas las modalidades tienen prefijo)
// v9.0.4: Añade prefijos code./cortex. a tensores según bloque
// v9.0.3: Parchea vocab_size desde tensor real
//
// ============================================================================

use std::path::Path;
use anyhow::{Result, Context};

use crate::hqs::{self, QuantFormat};
use crate::hnf::HnfWriter;
use crate::mapping::{ModelMapper, BlockType, create_mapper};
use crate::safetensor::SafetensorReader;

/// Estadísticas de conversión
#[derive(Debug, Default)]
pub struct BuildStats {
    pub fp16_count: usize,
    pub hq5k_count: usize,
    pub hq4k_count: usize,
    pub skipped_count: usize,
    pub total_bytes: usize,
}

impl BuildStats {
    pub fn total_tensors(&self) -> usize {
        self.fp16_count + self.hq5k_count + self.hq4k_count
    }
    
    pub fn record(&mut self, format: QuantFormat, size: usize) {
        match format {
            QuantFormat::FP16 => self.fp16_count += 1,
            QuantFormat::HQ5K => self.hq5k_count += 1,
            QuantFormat::HQ4K => self.hq4k_count += 1,
            _ => {}
        }
        self.total_bytes += size;
    }
}

/// Resuelve el nombre final del tensor con prefijo según bloque.
/// 
/// v9.0.5: TODAS las modalidades llevan prefijo para consistencia:
/// - TextModel (0x0): prefijo "text."   → "text.layer0.attn.q_proj.weight"
/// - CodeExec (0x8): prefijo "code."   → "code.layer0.attn.q_proj.weight"
/// - Cortex (0x7):   prefijo "cortex." → "cortex.layer0.attn.q_proj.weight"
/// - Vision (0x1):   prefijo "vision." → "vision.layer0.attn.q_proj.weight"
/// - Audio (0x2):    prefijo "audio."  → "audio.layer0.attn.q_proj.weight"
fn resolve_tensor_name(canonical_name: &str, target_block: BlockType) -> String {
    match target_block {
        BlockType::TextModel => {
            // v9.0.5: Añadir prefijo "text." para consistencia
            if canonical_name.starts_with("text.") {
                canonical_name.to_string()
            } else {
                format!("text.{}", canonical_name)
            }
        }
        BlockType::CodeExec => {
            if canonical_name.starts_with("code.") {
                canonical_name.to_string()
            } else {
                format!("code.{}", canonical_name)
            }
        }
        BlockType::Cortex => {
            if canonical_name.starts_with("cortex.") {
                canonical_name.to_string()
            } else {
                format!("cortex.{}", canonical_name)
            }
        }
        BlockType::Vision => {
            if canonical_name.starts_with("vision.") {
                canonical_name.to_string()
            } else {
                format!("vision.{}", canonical_name)
            }
        }
        BlockType::Audio => {
            if canonical_name.starts_with("audio.") {
                canonical_name.to_string()
            } else {
                format!("audio.{}", canonical_name)
            }
        }
        _ => canonical_name.to_string(),
    }
}

/// Procesa un modelo y escribe al bloque especificado
pub fn process_model(
    model_path: &Path,
    target_block: BlockType,
    writer: &mut HnfWriter,
    default_quant: QuantFormat,
    use_mse: bool,
    verbose: bool,
) -> Result<BuildStats> {
    let mut stats = BuildStats::default();
    
    // Crear mapper para la arquitectura
    let mapper = create_mapper(model_path)
        .with_context(|| format!("Failed to create mapper for {}", model_path.display()))?;
    
    if verbose {
        println!("  Mapper: {}", mapper.name());
        println!("  Layers: {}", mapper.num_layers());
        println!("  Target block: {} (0x{:X})", target_block.name(), target_block.as_usize());
    }
    
    // Abrir safetensors
    let reader = SafetensorReader::from_folder(model_path)
        .with_context(|| format!("Failed to open model {}", model_path.display()))?;
    
    let total_tensors = reader.len();
    if verbose {
        println!("  Tensors: {}", total_tensors);
    }
    
    // Procesar cada tensor
    for (idx, (name, info)) in reader.iter_tensors().enumerate() {
        // El mapper decide nombre canónico y sugiere cuantización
        let mapping = match mapper.map_tensor(name) {
            Some(m) => m,
            None => {
                stats.skipped_count += 1;
                continue;
            }
        };
        
        // ═══════════════════════════════════════════════════════════════════
        // RESOLVER NOMBRE FINAL CON PREFIJO SEGÚN BLOQUE
        // ═══════════════════════════════════════════════════════════════════
        let final_name = resolve_tensor_name(&mapping.canonical_name, target_block);
        
        // Resolver cuantización (mapper sugiere, default resuelve)
        let quant = mapping.quant_hint.resolve(default_quant);
        
        // Leer datos
        let data = reader.read(name)?;
        
        // Cuantizar
        let quantized = hqs::quantize(&data, quant, use_mse);
        let quantized_size = quantized.len();
        
        // Escribir al bloque con nombre final (incluye prefijo si aplica)
        writer.write_tensor(
            target_block.as_usize(),
            &final_name,
            &quant.to_string().to_lowercase(),
            &info.shape,
            &quantized,
        )?;
        
        stats.record(quant, quantized_size);
        
        // Progress
        if verbose && (idx + 1) % 20 == 0 {
            println!("    [{}/{}] {}", idx + 1, total_tensors, final_name);
        }
    }
    
    // Finalizar bloque (calcula checksum)
    writer.finalize_block(target_block.as_usize())?;
    
    Ok(stats)
}

/// Escribe execution_hints combinados de múltiples mappers
/// v9.0.5: TEXT también va bajo "text" con "text_enabled" para consistencia
/// v9.0.3: Parchea vocab_size desde el tensor real token_embedding.weight
pub fn write_combined_hints(
    writer: &mut HnfWriter,
    mappers: &[(&dyn ModelMapper, BlockType)],
) -> Result<()> {
    let mut combined = serde_json::Map::new();
    
    // Obtener manifests para parchear vocab_size
    let manifests = writer.tensor_manifests();
    
    for (mapper, block) in mappers {
        let mut hints = mapper.execution_hints();
        
        // ═══════════════════════════════════════════════════════════════════
        // PARCHEAR vocab_size DESDE TENSOR REAL
        // ═══════════════════════════════════════════════════════════════════
        if let Some(obj) = hints.as_object_mut() {
            let block_idx = block.as_usize();
            
            // v9.0.5: Determinar nombre del embedding según bloque (con prefijo)
            let embedding_names: Vec<&str> = match block {
                BlockType::TextModel => vec!["text.token_embedding.weight", "token_embedding.weight"],
                BlockType::CodeExec => vec!["code.token_embedding.weight", "token_embedding.weight"],
                BlockType::Cortex => vec!["cortex.token_embedding.weight", "token_embedding.weight"],
                _ => vec!["token_embedding.weight"],
            };
            
            // Buscar token_embedding.weight en este bloque
            if let Some(tensors) = manifests.get(block_idx) {
                for t in tensors {
                    if embedding_names.contains(&t.name.as_str()) {
                        if let Some(&vocab) = t.shape.first() {
                            let old_vocab = obj.get("vocab_size")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0);
                            
                            if vocab != old_vocab as usize {
                                eprintln!(
                                    "[INFO] Patching vocab_size: {} -> {} (from tensor shape in {})",
                                    old_vocab, vocab, block.name()
                                );
                            }
                            
                            obj.insert(
                                "vocab_size".to_string(), 
                                serde_json::json!(vocab)
                            );
                        }
                        break;
                    }
                }
            }
        }
        
        // v9.0.5: Insertar hints - TODAS las modalidades usan el mismo patrón
        match block {
            BlockType::TextModel => {
                // v9.0.5: TEXT también va bajo "text" para consistencia
                combined.insert("text_enabled".to_string(), serde_json::Value::Bool(true));
                combined.insert("text".to_string(), hints);
            }
            BlockType::Vision => {
                combined.insert("vision_enabled".to_string(), serde_json::Value::Bool(true));
                combined.insert("vision".to_string(), hints);
            }
            BlockType::Audio => {
                combined.insert("audio_enabled".to_string(), serde_json::Value::Bool(true));
                combined.insert("audio".to_string(), hints);
            }
            BlockType::Cortex => {
                combined.insert("cortex_enabled".to_string(), serde_json::Value::Bool(true));
                combined.insert("cortex".to_string(), hints);
            }
            BlockType::CodeExec => {
                combined.insert("code_enabled".to_string(), serde_json::Value::Bool(true));
                combined.insert("code".to_string(), hints);
            }
            _ => {
                combined.insert(block.name().to_string(), hints);
            }
        }
    }
    
    writer.write_execution_hints(&serde_json::Value::Object(combined))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_resolve_tensor_name_text() {
        // v9.0.5: TEXT ahora tiene prefijo
        let name = resolve_tensor_name("layer0.attn.q_proj.weight", BlockType::TextModel);
        assert_eq!(name, "text.layer0.attn.q_proj.weight");
        
        // No duplicar si ya tiene prefijo
        let name2 = resolve_tensor_name("text.layer0.attn.q_proj.weight", BlockType::TextModel);
        assert_eq!(name2, "text.layer0.attn.q_proj.weight");
    }
    
    #[test]
    fn test_resolve_tensor_name_code() {
        let name = resolve_tensor_name("layer0.attn.q_proj.weight", BlockType::CodeExec);
        assert_eq!(name, "code.layer0.attn.q_proj.weight");
        
        // No duplicar si ya tiene prefijo
        let name2 = resolve_tensor_name("code.layer0.attn.q_proj.weight", BlockType::CodeExec);
        assert_eq!(name2, "code.layer0.attn.q_proj.weight");
    }
    
    #[test]
    fn test_resolve_tensor_name_cortex() {
        let name = resolve_tensor_name("layer0.mlp.gate.weight", BlockType::Cortex);
        assert_eq!(name, "cortex.layer0.mlp.gate.weight");
        
        // No duplicar si ya tiene prefijo
        let name2 = resolve_tensor_name("cortex.layer0.mlp.gate.weight", BlockType::Cortex);
        assert_eq!(name2, "cortex.layer0.mlp.gate.weight");
    }
    
    #[test]
    fn test_resolve_tensor_name_vision() {
        let name = resolve_tensor_name("layer0.attn.q_proj.weight", BlockType::Vision);
        assert_eq!(name, "vision.layer0.attn.q_proj.weight");
    }
    
    #[test]
    fn test_resolve_tensor_name_audio() {
        let name = resolve_tensor_name("layer0.attn.q_proj.weight", BlockType::Audio);
        assert_eq!(name, "audio.layer0.attn.q_proj.weight");
    }
}
