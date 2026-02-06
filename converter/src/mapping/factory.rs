// src/mapping/factory.rs
// ============================================================================
// MAPPER FACTORY - Crea el mapper correcto para cada arquitectura
// ============================================================================

use std::path::Path;
use anyhow::{Result, Context};
use serde_json::Value;

use super::traits::ModelMapper;
use super::qwen2::Qwen2Mapper;
use super::llama::LlamaMapper;
use super::clip::ClipMapper;
use super::phi::PhiMapper;  // AÑADIDO

/// Detecta la arquitectura de un modelo desde config.json
pub fn detect_architecture(config: &Value) -> String {
    // Por model_type
    if let Some(model_type) = config.get("model_type").and_then(|v| v.as_str()) {
        let mt = model_type.to_lowercase();
        
        // Vision encoders
        if mt.contains("clip") || mt.contains("siglip") {
            return "clip".to_string();
        }
        if mt.contains("vit") {
            return "vit".to_string();
        }
        
        // LLMs - Phi ANTES de Llama (phi3 contiene "phi")
        if mt.contains("phi") {
            return "phi".to_string();
        }
        if mt.contains("qwen") {
            return "qwen2".to_string();
        }
        if mt.contains("llama") || mt.contains("deepseek") || mt.contains("codellama") {
            return "llama".to_string();
        }
        if mt.contains("mistral") {
            return "mistral".to_string();
        }
        if mt.contains("gemma") {
            return "gemma".to_string();
        }
        
        return mt;
    }
    
    // Por architectures
    if let Some(archs) = config.get("architectures").and_then(|v| v.as_array()) {
        if let Some(arch) = archs.first().and_then(|v| v.as_str()) {
            let arch_lower = arch.to_lowercase();
            
            // Vision
            if arch_lower.contains("clip") || arch_lower.contains("siglip") {
                return "clip".to_string();
            }
            
            // Phi - detectar antes de Llama
            if arch_lower.contains("phi") {
                return "phi".to_string();
            }
            
            // Qwen
            if arch_lower.contains("qwen") {
                return "qwen2".to_string();
            }
            
            // Llama family (includes DeepSeek, CodeLlama, etc.)
            if arch_lower.contains("llama") 
                || arch_lower.contains("deepseek")
                || arch_lower.contains("mistral") {
                return "llama".to_string();
            }
            
            // Gemma
            if arch_lower.contains("gemma") {
                return "gemma".to_string();
            }
        }
    }
    
    "generic".to_string()
}

/// Lee config.json de un modelo
pub fn load_config(model_path: &Path) -> Result<Value> {
    let config_path = model_path.join("config.json");
    
    if !config_path.exists() {
        anyhow::bail!("No config.json found in {}", model_path.display());
    }
    
    let data = std::fs::read_to_string(&config_path)
        .with_context(|| format!("Failed to read {}", config_path.display()))?;
    
    let config: Value = serde_json::from_str(&data)
        .with_context(|| "Invalid JSON in config.json")?;
    
    Ok(config)
}

/// Crea el mapper correcto para un modelo
pub fn create_mapper(model_path: &Path) -> Result<Box<dyn ModelMapper>> {
    let config = load_config(model_path)?;
    let arch = detect_architecture(&config);
    
    println!("[INFO] Detected architecture: {}", arch);
    
    match arch.as_str() {
        "qwen2" | "qwen" => {
            Ok(Box::new(Qwen2Mapper::from_json(&config)))
        }
        
        "llama" | "mistral" | "deepseek" | "codellama" => {
            Ok(Box::new(LlamaMapper::from_json(&config)))
        }
        
        "clip" | "siglip" | "vit" => {
            Ok(Box::new(ClipMapper::from_json(&config)))
        }
        
        // AÑADIDO: Phi family
        "phi" | "phi3" | "phi4" => {
            Ok(Box::new(PhiMapper::from_json(&config)))
        }
        
        // TODO: Añadir más arquitecturas
        // "gemma" | "gemma2" => Ok(Box::new(GemmaMapper::from_json(&config))),
        // "whisper" => Ok(Box::new(WhisperMapper::from_json(&config))),
        
        _ => {
            eprintln!("[WARN] Unknown architecture '{}', trying llama mapper", arch);
            Ok(Box::new(LlamaMapper::from_json(&config)))
        }
    }
}
