// src/bin/validate.rs
// ============================================================================
// HELIOS FORMAT VALIDATOR - Validación Total
// ============================================================================
//
// No pasa ni un pelo de mosca.
//
// Valida:
//   - HNFv9 (.hnf) - Modelo principal
//   - HTF v1.2.1 (.htf embebido) - Tokenizer
//
// Uso:
//   helios-validate archivo.hnf [-v]
//
// ============================================================================

use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

use clap::Parser;

// ============================================================================
// CONSTANTES HNFv9 (HNFv9_MASTER_SPEC.txt)
// ============================================================================

const HNF_MAGIC: &[u8; 8] = b"HNFv9\x00\x00\x00";
const HNF_VERSION_MAJOR: u16 = 9;
const HNF_BLOCK_COUNT: usize = 16;
const HNF_HEADER_SIZE: usize = 64;
const HNF_BLOCK_ENTRY_SIZE: usize = 32;
const HNF_BLOCK_TABLE_SIZE: usize = HNF_BLOCK_COUNT * HNF_BLOCK_ENTRY_SIZE; // 512
const HNF_BLOCK_TABLE_OFFSET: usize = HNF_HEADER_SIZE; // 64
const HNF_ALIGNMENT: usize = 32; // CUDA alignment

// Nombres de bloques según spec (0x7 = cortex, 0x9 = tokenizer)
const HNF_BLOCK_NAMES: [&str; 16] = [
    "text_model",       // 0x0 - OBLIGATORIO
    "vision",           // 0x1
    "audio",            // 0x2
    "video",            // 0x3
    "spatial_3d",       // 0x4
    "personality",      // 0x5 - MAX 20MB
    "memory",           // 0x6 - MAX 50MB
    "cortex",           // 0x7
    "code_exec",        // 0x8
    "tokenizer",        // 0x9 - HTF tokenizer
    "execution_hints",  // 0xA - OBLIGATORIO
    "expert_router",    // 0xB
    "tools",            // 0xC
    "reserved_1",       // 0xD
    "reserved_2",       // 0xE
    "reserved_3",       // 0xF
];

// Límites
const PERSONALITY_MAX_SIZE: usize = 20 * 1024 * 1024; // 20 MB
const MEMORY_MAX_SIZE: usize = 50 * 1024 * 1024;      // 50 MB

// ============================================================================
// CONSTANTES HTF v1.2.1 (HTF_v1_2_1_SPEC.txt)
// ============================================================================

const HTF_MAGIC_V2: &[u8; 4] = b"HTF2";
const HTF_HEADER_SIZE: usize = 32;
const HTF_DOMAIN_ENTRY_SIZE: usize = 32;
const HTF_MAX_DOMAINS: u8 = 8;
const HTF_V2_VALID_VERSIONS: [u16; 2] = [0x0102, 0x0103]; // v1.2.0 y v1.2.1

// Domain types (§5)
const HTF_DOMAIN_TEXT: u8 = 0x00;
const HTF_DOMAIN_VISION: u8 = 0x01;
const HTF_DOMAIN_AUDIO: u8 = 0x02;
const HTF_DOMAIN_CODE: u8 = 0x03;

// Domain flags (§6) - CORRECTOS según spec
const HTF_FLAG_HAS_VOCAB: u8 = 0x01;      // bit 0
const HTF_FLAG_HAS_CODEBOOK: u8 = 0x02;   // bit 1
const HTF_FLAG_HAS_MERGES: u8 = 0x04;     // bit 2
const HTF_FLAG_IS_PRIMARY: u8 = 0x08;     // bit 3
const HTF_FLAG_SHARED_SPECIAL: u8 = 0x10; // bit 4

// ============================================================================
// CONSTANTES EXECUTION_HINTS v1.2 (EXECUTION_HINTS_v1_2_SPEC.txt)
// ============================================================================

// Campos obligatorios según spec
const EXEC_HINTS_REQUIRED: &[&str] = &[
    "arch",
    "dtype",
    "num_hidden_layers",
    "hidden_size",
    "intermediate_size",
    "vocab_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "attention_type",
    "mlp_type",
    "mlp_activation",  // ← CORREGIDO (era "hidden_act")
    "norm_type",
];

// Valores válidos según spec
const VALID_ARCHS: &[&str] = &[
    "llama", "llama2", "llama3",
    "qwen", "qwen2",
    "gemma", "gemma2",
    "phi", "phi3", "phi4",
    "mistral", "mixtral",
    "falcon", "mpt", "gpt2",
    "clip", "siglip", "vit",
];

const VALID_DTYPES: &[&str] = &["fp16", "bf16", "fp32"];
const VALID_ATTENTION_TYPES: &[&str] = &["mha", "gqa", "mqa"];
const VALID_MLP_TYPES: &[&str] = &["swiglu", "geglu", "gated", "standard"];
const VALID_MLP_ACTIVATIONS: &[&str] = &["silu", "gelu", "gelu_new", "gelu_fast", "relu", "quick_gelu"];
const VALID_NORM_TYPES: &[&str] = &["rmsnorm", "layernorm"];

// ============================================================================
// ESTRUCTURAS
// ============================================================================

#[derive(Debug)]
struct ValidationError {
    category: String,
    message: String,
    fatal: bool,
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let prefix = if self.fatal { "FATAL" } else { "WARN" };
        write!(f, "[{}] {}: {}", prefix, self.category, self.message)
    }
}

#[derive(Default)]
struct ValidationResult {
    errors: Vec<ValidationError>,
    header: Option<HnfHeader>,
    blocks: Vec<BlockEntry>,
    execution_hints: Option<serde_json::Value>,
    htf_info: Option<HtfInfo>,
    manifest: Option<serde_json::Value>,
}

impl ValidationResult {
    fn is_valid(&self) -> bool {
        !self.errors.iter().any(|e| e.fatal)
    }
    
    fn fatal_count(&self) -> usize {
        self.errors.iter().filter(|e| e.fatal).count()
    }
    
    fn warn_count(&self) -> usize {
        self.errors.iter().filter(|e| !e.fatal).count()
    }
    
    fn add_error(&mut self, category: &str, message: &str, fatal: bool) {
        self.errors.push(ValidationError {
            category: category.to_string(),
            message: message.to_string(),
            fatal,
        });
    }
}

#[derive(Debug, Clone)]
struct HnfHeader {
    magic: [u8; 8],
    version_major: u16,
    version_minor: u16,
    flags: u32,
    block_count: u32,
    header_size: u32,
    block_table_offset: u64,
    manifest_offset: u64,
    manifest_size: u64,
    file_size: u64,
    checksum: u32,
}

#[derive(Debug, Clone)]
struct BlockEntry {
    id: u32,
    block_type: u32,
    offset: u64,
    size: u64,
    checksum: u64,
    name: String,
}

#[derive(Debug, Clone)]
struct HtfInfo {
    offset: usize,
    size: usize,
    version: u16,
    num_domains: u8,
    domains: Vec<HtfDomain>,
}

#[derive(Debug, Clone)]
struct HtfDomain {
    domain_type: u8,
    flags: u8,
    vocab_size: u32,
    data_offset: u64,
    data_size: u64,
}

// ============================================================================
// UTILIDADES
// ============================================================================

fn format_size(size: usize) -> String {
    if size >= 1024 * 1024 * 1024 {
        format!("{:.2} GB", size as f64 / 1024.0 / 1024.0 / 1024.0)
    } else if size >= 1024 * 1024 {
        format!("{:.2} MB", size as f64 / 1024.0 / 1024.0)
    } else if size >= 1024 {
        format!("{:.2} KB", size as f64 / 1024.0)
    } else {
        format!("{} bytes", size)
    }
}

fn read_u16_le(data: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([data[offset], data[offset + 1]])
}

fn read_u32_le(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        data[offset], data[offset + 1], 
        data[offset + 2], data[offset + 3]
    ])
}

fn read_u64_le(data: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes([
        data[offset], data[offset + 1], data[offset + 2], data[offset + 3],
        data[offset + 4], data[offset + 5], data[offset + 6], data[offset + 7],
    ])
}

fn xxh3_64(data: &[u8]) -> u64 {
    xxhash_rust::xxh3::xxh3_64(data)
}

fn domain_type_name(t: u8) -> &'static str {
    match t {
        HTF_DOMAIN_TEXT => "TEXT",
        HTF_DOMAIN_VISION => "VISION",
        HTF_DOMAIN_AUDIO => "AUDIO",
        HTF_DOMAIN_CODE => "CODE",
        _ => "UNKNOWN",
    }
}

fn domain_canonical_name(t: u8) -> &'static [u8] {
    match t {
        HTF_DOMAIN_TEXT => b"text",
        HTF_DOMAIN_VISION => b"vision",
        HTF_DOMAIN_AUDIO => b"audio",
        HTF_DOMAIN_CODE => b"code",
        _ => b"unknown",
    }
}

// ============================================================================
// VALIDADOR HNF
// ============================================================================

struct HnfValidator {
    data: Vec<u8>,
    verbose: bool,
    result: ValidationResult,
}

impl HnfValidator {
    fn new(data: Vec<u8>, verbose: bool) -> Self {
        Self {
            data,
            verbose,
            result: ValidationResult::default(),
        }
    }
    
    fn log(&self, msg: &str) {
        if self.verbose {
            println!("    {}", msg);
        }
    }
    
    fn validate(mut self) -> ValidationResult {
        println!("\n{}", "=".repeat(72));
        println!("HNFv9 STRICT VALIDATOR");
        println!("{}", "=".repeat(72));
        println!("  Tamaño: {}", format_size(self.data.len()));
        
        // Lista de validaciones
        let checks: Vec<(&str, fn(&mut Self))> = vec![
            ("[1/12] HEADER", Self::validate_header),
            ("[2/12] BLOCK TABLE", Self::validate_block_table),
            ("[3/12] BLOQUES OBLIGATORIOS", Self::validate_required_blocks),
            ("[4/12] LÍMITES DE TAMAÑO", Self::validate_block_limits),
            ("[5/12] FLAGS COHERENTES", Self::validate_flags),
            ("[6/12] ORDEN FÍSICO", Self::validate_physical_order),
            ("[7/12] ALINEACIÓN", Self::validate_alignment),
            ("[8/12] EXECUTION_HINTS", Self::validate_execution_hints),
            ("[9/12] TOKENIZER HTF", Self::validate_tokenizer),
            ("[10/12] MANIFEST", Self::validate_manifest),
            ("[11/12] CHECKSUMS", Self::validate_checksums),
            ("[12/12] TENSORES", Self::validate_tensors),
        ];
        
        for (name, check_fn) in checks {
            println!("\n{}", "─".repeat(72));
            println!("{}", name);
            check_fn(&mut self);
        }
        
        self.print_summary();
        self.result
    }
    
    fn validate_header(&mut self) {
        if self.data.len() < HNF_HEADER_SIZE {
            self.result.add_error("HEADER", 
                &format!("Archivo muy pequeño: {} < {}", self.data.len(), HNF_HEADER_SIZE), true);
            return;
        }
        
        let mut magic = [0u8; 8];
        magic.copy_from_slice(&self.data[0..8]);
        
        let header = HnfHeader {
            magic,
            version_major: read_u16_le(&self.data, 8),
            version_minor: read_u16_le(&self.data, 10),
            flags: read_u32_le(&self.data, 12),
            block_count: read_u32_le(&self.data, 16),
            header_size: read_u32_le(&self.data, 20),
            block_table_offset: read_u64_le(&self.data, 24),
            manifest_offset: read_u64_le(&self.data, 32),
            manifest_size: read_u64_le(&self.data, 40),
            file_size: read_u64_le(&self.data, 48),
            checksum: read_u32_le(&self.data, 56),
        };
        
        // Validaciones estrictas
        if &header.magic != HNF_MAGIC {
            self.result.add_error("HEADER", 
                &format!("Magic inválido: {:?} (esperado: {:?})", header.magic, HNF_MAGIC), true);
        } else {
            self.log(&format!("✓ Magic: {:?}", header.magic));
        }
        
        if header.version_major != HNF_VERSION_MAJOR {
            self.result.add_error("HEADER",
                &format!("version_major: {} (esperado: {})", header.version_major, HNF_VERSION_MAJOR), true);
        } else {
            self.log(&format!("✓ Versión: {}.{}", header.version_major, header.version_minor));
        }
        
        if header.block_count != HNF_BLOCK_COUNT as u32 {
            self.result.add_error("HEADER",
                &format!("block_count: {} (esperado: {})", header.block_count, HNF_BLOCK_COUNT), true);
        } else {
            self.log(&format!("✓ block_count: {}", header.block_count));
        }
        
        if header.header_size != HNF_HEADER_SIZE as u32 {
            self.result.add_error("HEADER",
                &format!("header_size: {} (esperado: {})", header.header_size, HNF_HEADER_SIZE), true);
        }
        
        if header.block_table_offset != HNF_HEADER_SIZE as u64 {
            self.result.add_error("HEADER",
                &format!("block_table_offset: {} (esperado: {})", header.block_table_offset, HNF_HEADER_SIZE), true);
        }
        
        if header.file_size != self.data.len() as u64 {
            self.result.add_error("HEADER",
                &format!("file_size: {} (actual: {})", header.file_size, self.data.len()), true);
        } else {
            self.log(&format!("✓ file_size: {}", header.file_size));
        }
        
        if header.manifest_offset + header.manifest_size != header.file_size {
            self.result.add_error("HEADER",
                &format!("Manifest no está al EOF: {}+{} != {}", 
                    header.manifest_offset, header.manifest_size, header.file_size), true);
        } else {
            self.log(&format!("✓ Manifest al EOF: offset={}, size={}", 
                header.manifest_offset, header.manifest_size));
        }
        
        self.result.header = Some(header);
    }
    
    fn validate_block_table(&mut self) {
        if self.data.len() < HNF_BLOCK_TABLE_OFFSET + HNF_BLOCK_TABLE_SIZE {
            self.result.add_error("BLOCK_TABLE", "Archivo muy pequeño para Block Table", true);
            return;
        }
        
        let mut blocks = Vec::new();
        
        for i in 0..HNF_BLOCK_COUNT {
            let offset = HNF_BLOCK_TABLE_OFFSET + i * HNF_BLOCK_ENTRY_SIZE;
            
            let block = BlockEntry {
                id: read_u32_le(&self.data, offset),
                block_type: read_u32_le(&self.data, offset + 4),
                offset: read_u64_le(&self.data, offset + 8),
                size: read_u64_le(&self.data, offset + 16),
                checksum: read_u64_le(&self.data, offset + 24),
                name: HNF_BLOCK_NAMES[i].to_string(),
            };
            
            // Validar id y type
            if block.id != i as u32 {
                self.result.add_error("BLOCK_TABLE",
                    &format!("Bloque {}: block_id={} (esperado: {})", i, block.id, i), true);
            }
            
            if block.block_type != i as u32 {
                self.result.add_error("BLOCK_TABLE",
                    &format!("Bloque {}: block_type={} (esperado: {})", i, block.block_type, i), true);
            }
            
            // Bloque vacío debe tener checksum 0
            if block.size == 0 && block.checksum != 0 {
                self.result.add_error("BLOCK_TABLE",
                    &format!("Bloque {} vacío con checksum != 0", i), false);
            }
            
            if block.size > 0 {
                self.log(&format!("✓ [{:2}] {:20}: {:>12} @ {}", 
                    i, HNF_BLOCK_NAMES[i], format_size(block.size as usize), block.offset));
            }
            
            blocks.push(block);
        }
        
        self.result.blocks = blocks;
    }
    
    fn validate_required_blocks(&mut self) {
        if self.result.blocks.is_empty() {
            return;
        }
        
        // text_model (índice 0) - OBLIGATORIO
        if self.result.blocks[0].size == 0 {
            self.result.add_error("REQUIRED", "text_model (bloque 0) está VACÍO - OBLIGATORIO", true);
        } else {
            self.log(&format!("✓ text_model: {}", format_size(self.result.blocks[0].size as usize)));
        }
        
        // execution_hints (índice 10) - OBLIGATORIO
        if self.result.blocks[10].size == 0 {
            self.result.add_error("REQUIRED", "execution_hints (bloque 10) está VACÍO - OBLIGATORIO", true);
        } else {
            self.log(&format!("✓ execution_hints: {}", format_size(self.result.blocks[10].size as usize)));
        }
    }
    
    fn validate_block_limits(&mut self) {
        if self.result.blocks.is_empty() {
            return;
        }
        
        // personality (índice 5) <= 20MB
        if self.result.blocks[5].size > PERSONALITY_MAX_SIZE as u64 {
            self.result.add_error("LIMITS",
                &format!("personality excede 20MB: {}", format_size(self.result.blocks[5].size as usize)), true);
        } else if self.result.blocks[5].size > 0 {
            self.log(&format!("✓ personality: {} (≤ 20MB)", format_size(self.result.blocks[5].size as usize)));
        }
        
        // memory (índice 6) <= 50MB
        if self.result.blocks[6].size > MEMORY_MAX_SIZE as u64 {
            self.result.add_error("LIMITS",
                &format!("memory excede 50MB: {}", format_size(self.result.blocks[6].size as usize)), true);
        } else if self.result.blocks[6].size > 0 {
            self.log(&format!("✓ memory: {} (≤ 50MB)", format_size(self.result.blocks[6].size as usize)));
        }
    }
    
    fn validate_flags(&mut self) {
        let header = match &self.result.header {
            Some(h) => h.clone(),
            None => return,
        };
        
        if self.result.blocks.is_empty() {
            return;
        }
        
        let flags = header.flags;
        
        // Mapeo flag -> índice de bloque
        let flag_block_map: [(u32, usize, &str); 10] = [
            (1 << 0, 1, "vision"),
            (1 << 1, 2, "audio"),
            (1 << 2, 3, "video"),
            (1 << 3, 4, "spatial_3d"),
            (1 << 4, 5, "personality"),
            (1 << 5, 6, "memory"),
            (1 << 6, 7, "cortex"),  // ← CORREGIDO
            (1 << 7, 8, "code_exec"),
            (1 << 8, 9, "tools"),
            (1 << 9, 11, "expert_router"),
        ];
        
        for (flag, idx, name) in flag_block_map.iter() {
            let has_flag = (flags & flag) != 0;
            let has_data = self.result.blocks[*idx].size > 0;
            
            if has_flag && !has_data {
                self.result.add_error("FLAGS",
                    &format!("Flag {} activo pero bloque vacío", name), false);
            } else if !has_flag && has_data {
                self.result.add_error("FLAGS",
                    &format!("Bloque {} tiene datos pero flag inactivo", name), false);
            } else if has_flag && has_data {
                self.log(&format!("✓ {}: flag y datos coherentes", name));
            }
        }
        
        // IS_MULTIMODAL
        let is_multimodal = (flags & (1 << 11)) != 0;
        let has_multimodal = self.result.blocks[1].size > 0 
            || self.result.blocks[2].size > 0 
            || self.result.blocks[3].size > 0;
        
        if is_multimodal && !has_multimodal {
            self.result.add_error("FLAGS", "IS_MULTIMODAL activo pero no hay datos multimodales", false);
        }
    }
    
    fn validate_physical_order(&mut self) {
        if self.result.blocks.is_empty() {
            return;
        }
        
        let header = match &self.result.header {
            Some(h) => h.clone(),
            None => return,
        };
        
        // Clone para evitar borrow conflict
        let blocks = self.result.blocks.clone();
        let mut prev_end = (HNF_HEADER_SIZE + HNF_BLOCK_TABLE_SIZE) as u64;
        
        for (i, block) in blocks.iter().enumerate() {
            if block.size == 0 {
                continue;
            }
            
            if block.offset < prev_end {
                self.result.add_error("ORDER",
                    &format!("Bloque {}: offset {} < fin anterior {}", i, block.offset, prev_end), true);
            }
            
            let gap = block.offset - prev_end;
            if gap > HNF_ALIGNMENT as u64 {
                self.result.add_error("ORDER",
                    &format!("Bloque {}: hueco de {} bytes (max {})", i, gap, HNF_ALIGNMENT), false);
            }
            
            prev_end = block.offset + block.size;
        }
        
        if header.manifest_offset > 0 && header.manifest_offset < prev_end {
            self.result.add_error("ORDER",
                &format!("Manifest offset {} antes del fin de bloques {}", header.manifest_offset, prev_end), true);
        } else {
            self.log(&format!("✓ Orden correcto: último bloque termina en {}, manifest en {}", 
                prev_end, header.manifest_offset));
        }
    }
    
    fn validate_alignment(&mut self) {
        if self.result.blocks.is_empty() {
            return;
        }
        
        // Clone para evitar borrow conflict
        let blocks = self.result.blocks.clone();
        let mut aligned = 0;
        let mut total = 0;
        
        for (i, block) in blocks.iter().enumerate() {
            if block.size == 0 {
                continue;
            }
            
            total += 1;
            if block.offset % HNF_ALIGNMENT as u64 == 0 {
                aligned += 1;
            } else {
                self.result.add_error("ALIGNMENT",
                    &format!("Bloque {}: offset {} NO alineado a {} bytes", i, block.offset, HNF_ALIGNMENT), false);
            }
        }
        
        self.log(&format!("✓ {}/{} bloques alineados a {} bytes", aligned, total, HNF_ALIGNMENT));
    }
    
    fn validate_execution_hints(&mut self) {
        if self.result.blocks.is_empty() || self.result.blocks[10].size == 0 {
            return;
        }
        
        let block = &self.result.blocks[10];
        let start = block.offset as usize;
        let end = start + block.size as usize;
        
        if end > self.data.len() {
            self.result.add_error("EXEC_HINTS", "Bloque fuera de límites", true);
            return;
        }
        
        let hints_data = &self.data[start..end];
        
        let hints: serde_json::Value = match serde_json::from_slice(hints_data) {
            Ok(v) => v,
            Err(e) => {
                self.result.add_error("EXEC_HINTS", &format!("JSON inválido: {}", e), true);
                return;
            }
        };
        
        self.log(&format!("✓ JSON válido ({} bytes)", block.size));
        
        // Campos obligatorios según EXECUTION_HINTS_v1_2_SPEC.txt
        let missing: Vec<&str> = EXEC_HINTS_REQUIRED.iter()
            .filter(|f| hints.get(*f).is_none())
            .copied()
            .collect();
        
        if !missing.is_empty() {
            self.result.add_error("EXEC_HINTS",
                &format!("Campos obligatorios faltantes: {:?}", missing), true);
        } else {
            self.log("✓ Campos obligatorios presentes");
        }
        
        // Validar valores permitidos
        if let Some(arch) = hints.get("arch").and_then(|v| v.as_str()) {
            if !VALID_ARCHS.contains(&arch) {
                self.result.add_error("EXEC_HINTS",
                    &format!("arch inválido: '{}' (válidos: {:?})", arch, VALID_ARCHS), false);
            }
            self.log(&format!("  arch: {}", arch));
        }
        
        if let Some(dtype) = hints.get("dtype").and_then(|v| v.as_str()) {
            if !VALID_DTYPES.contains(&dtype) {
                self.result.add_error("EXEC_HINTS",
                    &format!("dtype inválido: '{}' (válidos: {:?})", dtype, VALID_DTYPES), true);
            }
            self.log(&format!("  dtype: {}", dtype));
        }
        
        if let Some(attn) = hints.get("attention_type").and_then(|v| v.as_str()) {
            if !VALID_ATTENTION_TYPES.contains(&attn) {
                self.result.add_error("EXEC_HINTS",
                    &format!("attention_type inválido: '{}' (válidos: {:?})", attn, VALID_ATTENTION_TYPES), true);
            }
        }
        
        if let Some(mlp) = hints.get("mlp_type").and_then(|v| v.as_str()) {
            if !VALID_MLP_TYPES.contains(&mlp) {
                self.result.add_error("EXEC_HINTS",
                    &format!("mlp_type inválido: '{}' (válidos: {:?})", mlp, VALID_MLP_TYPES), true);
            }
        }
        
        if let Some(act) = hints.get("mlp_activation").and_then(|v| v.as_str()) {
            if !VALID_MLP_ACTIVATIONS.contains(&act) {
                self.result.add_error("EXEC_HINTS",
                    &format!("mlp_activation inválido: '{}' (válidos: {:?})", act, VALID_MLP_ACTIVATIONS), true);
            }
        }
        
        if let Some(norm) = hints.get("norm_type").and_then(|v| v.as_str()) {
            if !VALID_NORM_TYPES.contains(&norm) {
                self.result.add_error("EXEC_HINTS",
                    &format!("norm_type inválido: '{}' (válidos: {:?})", norm, VALID_NORM_TYPES), true);
            }
        }
        
        // Validar coherencia GQA
        if let (Some(n_heads), Some(n_kv_heads)) = (
            hints.get("num_attention_heads").and_then(|v| v.as_u64()),
            hints.get("num_key_value_heads").and_then(|v| v.as_u64()),
        ) {
            if n_kv_heads > n_heads {
                self.result.add_error("EXEC_HINTS",
                    &format!("num_key_value_heads ({}) > num_attention_heads ({})", n_kv_heads, n_heads), true);
            }
            if n_heads % n_kv_heads != 0 {
                self.result.add_error("EXEC_HINTS",
                    &format!("num_attention_heads ({}) no es divisible por num_key_value_heads ({})", n_heads, n_kv_heads), false);
            }
        }
        
        // Validar MoE
        if hints.get("moe_enabled").and_then(|v| v.as_bool()).unwrap_or(false) {
            if hints.get("num_experts").is_none() {
                self.result.add_error("EXEC_HINTS", "moe_enabled pero falta num_experts", true);
            }
            if hints.get("num_experts_per_tok").is_none() {
                self.result.add_error("EXEC_HINTS", "moe_enabled pero falta num_experts_per_tok", true);
            }
            if let (Some(n), Some(k)) = (
                hints.get("num_experts").and_then(|v| v.as_u64()),
                hints.get("num_experts_per_tok").and_then(|v| v.as_u64()),
            ) {
                self.log(&format!("  MoE: {} expertos, top-{}", n, k));
            }
        }
        
        // Validar multimodal
        if let Some(vision) = hints.get("vision_config") {
            self.log("  vision_config: presente");
            // Validar campos de vision
            if let Some(encoder) = vision.get("encoder_arch").and_then(|v| v.as_str()) {
                self.log(&format!("    encoder_arch: {}", encoder));
            }
        }
        
        self.result.execution_hints = Some(hints);
    }
    
    fn validate_tokenizer(&mut self) {
        let header = match &self.result.header {
            Some(h) => h.clone(),
            None => return,
        };
        
        if self.result.blocks.is_empty() {
            return;
        }
        
        // Encontrar fin del último bloque
        let mut last_end = (HNF_HEADER_SIZE + HNF_BLOCK_TABLE_SIZE) as u64;
        for block in &self.result.blocks {
            if block.size > 0 {
                let end = block.offset + block.size;
                if end > last_end {
                    last_end = end;
                }
            }
        }
        
        // Alinear a 32 bytes
        last_end = (last_end + HNF_ALIGNMENT as u64 - 1) & !(HNF_ALIGNMENT as u64 - 1);
        
        let tokenizer_size = header.manifest_offset.saturating_sub(last_end) as usize;
        let tokenizer_offset = last_end as usize;
        
        if tokenizer_size == 0 {
            self.result.add_error("TOKENIZER", "No hay espacio para tokenizer", false);
            return;
        }
        
        self.log(&format!("  Tokenizer: offset {}, size {}", tokenizer_offset, format_size(tokenizer_size)));
        
        if tokenizer_offset + 4 > self.data.len() {
            self.result.add_error("TOKENIZER", "Tokenizer fuera de límites", true);
            return;
        }
        
        let magic = &self.data[tokenizer_offset..tokenizer_offset + 4];
        
        if magic == HTF_MAGIC_V2 {
            self.log("✓ HTF v2.x (Multi-Domain) detectado");
            self.validate_htf_v2(tokenizer_offset, tokenizer_size);
        } else if &magic[0..3] == b"HTF" {
            self.log("✓ HTF v1.x detectado");
            // HTF v1 legacy - validación básica
        } else {
            self.result.add_error("TOKENIZER", &format!("Magic HTF inválido: {:?}", magic), true);
        }
    }
    
    fn validate_htf_v2(&mut self, offset: usize, size: usize) {
        if size < HTF_HEADER_SIZE {
            self.result.add_error("HTF", &format!("HTF v2 muy pequeño: {} < {}", size, HTF_HEADER_SIZE), true);
            return;
        }
        
        if offset + size > self.data.len() {
            self.result.add_error("HTF", "HTF fuera de límites del archivo", true);
            return;
        }
        
        let blob = &self.data[offset..offset + size];
        
        // Parse header
        let version = read_u16_le(blob, 4);
        let _flags = read_u16_le(blob, 6);
        let num_domains = blob[8];
        let reserved = &blob[9..16];
        let total_size = read_u64_le(blob, 16);
        let checksum = read_u64_le(blob, 24);
        
        self.log(&format!("  HTF v2 version: 0x{:04X}", version));
        self.log(&format!("  num_domains: {}", num_domains));
        
        // Validar version
        if !HTF_V2_VALID_VERSIONS.contains(&version) {
            self.result.add_error("HTF",
                &format!("Versión HTF inválida: 0x{:04X} (esperado 0x0102 o 0x0103)", version), true);
            return;
        }
        
        // Reserved debe ser cero
        if reserved.iter().any(|&b| b != 0) {
            self.result.add_error("HTF", "Header.reserved no es cero (archivo no contractual)", true);
            return;
        }
        
        // num_domains válido
        if num_domains == 0 || num_domains > HTF_MAX_DOMAINS {
            self.result.add_error("HTF",
                &format!("num_domains inválido: {} (1-{})", num_domains, HTF_MAX_DOMAINS), true);
            return;
        }
        
        // total_size CONTRACTUAL
        if total_size != size as u64 {
            self.result.add_error("HTF",
                &format!("total_size {} != tamaño real {}", total_size, size), true);
            return;
        }
        
        // Checksum CONTRACTUAL - Regla 6 de HTF spec
        // Input: header[0:24] + [0x00 × 8] + domain_table + todos los dominios
        let mut hasher = xxhash_rust::xxh3::Xxh3::new();
        hasher.update(&blob[..24]);
        hasher.update(&[0u8; 8]);
        hasher.update(&blob[HTF_HEADER_SIZE..]);
        let expected_checksum = hasher.digest();
        
        if checksum != expected_checksum {
            self.result.add_error("HTF",
                &format!("checksum inválido: 0x{:016X} != 0x{:016X}", checksum, expected_checksum), true);
            return;
        }
        
        self.log(&format!("✓ Checksum válido: 0x{:016X}", checksum));
        
        // Validar domain table
        let domain_table_size = num_domains as usize * HTF_DOMAIN_ENTRY_SIZE;
        let table_off = HTF_HEADER_SIZE;
        
        if table_off + domain_table_size > size {
            self.result.add_error("HTF", "Domain table fuera de límites", true);
            return;
        }
        
        let mut primary_count = 0;
        let mut expected_data_off = (HTF_HEADER_SIZE + domain_table_size) as u64;
        let mut domains = Vec::new();
        
        for i in 0..num_domains as usize {
            let entry_offset = table_off + i * HTF_DOMAIN_ENTRY_SIZE;
            let entry = &blob[entry_offset..entry_offset + HTF_DOMAIN_ENTRY_SIZE];
            
            let domain_type = entry[0];
            let domain_flags = entry[1];
            let reserved2 = &entry[2..4];
            let vocab_size = read_u32_le(entry, 4);
            let data_offset = read_u64_le(entry, 8);
            let data_size = read_u64_le(entry, 16);
            let name_hash = read_u64_le(entry, 24);
            
            // Reserved debe ser cero
            if reserved2 != [0, 0] {
                self.result.add_error("HTF", &format!("Domain[{}].reserved != 0", i), true);
                return;
            }
            
            // Tipo válido
            if domain_type > HTF_DOMAIN_CODE {
                self.result.add_error("HTF", &format!("Domain[{}] type inválido: {}", i, domain_type), true);
                return;
            }
            
            // Contar primarios
            if domain_flags & HTF_FLAG_IS_PRIMARY != 0 {
                primary_count += 1;
            }
            
            // Validar name_hash
            let expected_name = domain_canonical_name(domain_type);
            let expected_hash = xxh3_64(expected_name);
            if name_hash != expected_hash {
                self.result.add_error("HTF",
                    &format!("Domain[{}] name_hash inválido: 0x{:016X} != 0x{:016X}", i, name_hash, expected_hash), true);
                return;
            }
            
            // Alineamiento a 16 bytes para data_offset
            let aligned_expected = (expected_data_off + 15) & !15;
            if data_offset != aligned_expected {
                self.result.add_error("HTF",
                    &format!("Domain[{}] data_offset no alineado/contiguo: 0x{:X} != 0x{:X}", i, data_offset, aligned_expected), true);
                return;
            }
            
            // Verificar que no se sale del HTF
            if data_offset + data_size > size as u64 {
                self.result.add_error("HTF", &format!("Domain[{}] data fuera de límites", i), true);
                return;
            }
            
            // TEXT debe tener vocab
            if domain_type == HTF_DOMAIN_TEXT && vocab_size == 0 {
                self.result.add_error("HTF", "Domain TEXT sin vocabulario", true);
                return;
            }
            
            expected_data_off = data_offset + data_size;
            
            self.log(&format!("  Domain {}: {}, vocab={}, size={}", 
                i, domain_type_name(domain_type), vocab_size, format_size(data_size as usize)));
            
            domains.push(HtfDomain {
                domain_type,
                flags: domain_flags,
                vocab_size,
                data_offset,
                data_size,
            });
        }
        
        // Exactamente 1 primario (Regla 3)
        if primary_count != 1 {
            self.result.add_error("HTF",
                &format!("IS_PRIMARY inválido: se esperan 1, hay {}", primary_count), true);
            return;
        }
        
        // Con 1 domain, debe ser TEXT + PRIMARY (Regla 3)
        if num_domains == 1 {
            if domains[0].domain_type != HTF_DOMAIN_TEXT {
                self.result.add_error("HTF", "Con 1 domain, debe ser TEXT", true);
                return;
            }
            if domains[0].flags & HTF_FLAG_IS_PRIMARY == 0 {
                self.result.add_error("HTF", "Con 1 domain, debe tener IS_PRIMARY", true);
                return;
            }
        }
        
        self.result.htf_info = Some(HtfInfo {
            offset,
            size,
            version,
            num_domains,
            domains,
        });
    }
    
    fn validate_manifest(&mut self) {
        let header = match &self.result.header {
            Some(h) => h.clone(),
            None => return,
        };
        
        if header.manifest_size == 0 {
            self.result.add_error("MANIFEST", "Manifest vacío", true);
            return;
        }
        
        let start = header.manifest_offset as usize;
        let end = start + header.manifest_size as usize;
        
        if end > self.data.len() {
            self.result.add_error("MANIFEST", "Manifest fuera de límites", true);
            return;
        }
        
        let manifest_data = &self.data[start..end];
        
        let manifest: serde_json::Value = match serde_json::from_slice(manifest_data) {
            Ok(v) => v,
            Err(e) => {
                self.result.add_error("MANIFEST", &format!("JSON inválido: {}", e), true);
                return;
            }
        };
        
        self.log(&format!("✓ JSON válido ({} bytes)", header.manifest_size));
        
        // Campos esperados
        if manifest.get("format").is_none() {
            self.result.add_error("MANIFEST", "Campo 'format' faltante", true);
        } else if let Some(fmt) = manifest.get("format").and_then(|v| v.as_str()) {
            if !fmt.starts_with("HNFv") {
                self.result.add_error("MANIFEST", &format!("format inválido: {}", fmt), true);
            } else {
                self.log(&format!("  format: {}", fmt));
            }
        }
        
        if let Some(build) = manifest.get("build") {
            if let Some(converter) = build.get("converter").and_then(|v| v.as_str()) {
                self.log(&format!("  converter: {}", converter));
            }
            if let Some(ts) = build.get("timestamp").and_then(|v| v.as_str()) {
                self.log(&format!("  timestamp: {}", ts));
            }
        }
        
        self.result.manifest = Some(manifest);
    }
    
    fn validate_checksums(&mut self) {
        let header = match &self.result.header {
            Some(h) => h.clone(),
            None => return,
        };
        
        self.log(&format!("  Header CRC32: 0x{:08X}", header.checksum));
        
        // Clone para evitar borrow conflict
        let blocks = self.result.blocks.clone();
        
        // XXH3-64 por bloque
        let mut verified = 0;
        
        for (i, block) in blocks.iter().enumerate() {
            if block.size == 0 || block.checksum == 0 {
                continue;
            }
            
            let start = block.offset as usize;
            let end = start + block.size as usize;
            
            if end > self.data.len() {
                continue;
            }
            
            let block_data = &self.data[start..end];
            let calculated = xxh3_64(block_data);
            
            if calculated == block.checksum {
                verified += 1;
            } else {
                self.result.add_error("CHECKSUM",
                    &format!("Bloque {}: XXH3 esperado 0x{:016X}, calculado 0x{:016X}", 
                        i, block.checksum, calculated), true);
            }
        }
        
        if verified > 0 {
            self.log(&format!("✓ {} checksums XXH3-64 verificados", verified));
        }
    }
    
    fn validate_tensors(&mut self) {
        let manifest = match &self.result.manifest {
            Some(m) => m.clone(),
            None => return,
        };
        
        let tensors = match manifest.get("tensors").and_then(|v| v.as_array()) {
            Some(t) => t,
            None => return,
        };
        
        if tensors.is_empty() {
            return;
        }
        
        let required = ["name", "shape", "dtype", "offset", "size"];
        let mut errors = 0;
        
        for (i, tensor) in tensors.iter().enumerate() {
            let missing: Vec<&str> = required.iter()
                .filter(|f| tensor.get(*f).is_none())
                .copied()
                .collect();
            
            if !missing.is_empty() {
                if errors < 5 {
                    self.result.add_error("TENSORS",
                        &format!("Tensor {}: campos faltantes {:?}", i, missing), true);
                }
                errors += 1;
                continue;
            }
            
            // Validar offset dentro del archivo
            if let (Some(off), Some(sz)) = (
                tensor.get("offset").and_then(|v| v.as_u64()),
                tensor.get("size").and_then(|v| v.as_u64()),
            ) {
                if off + sz > self.data.len() as u64 {
                    let name = tensor.get("name").and_then(|v| v.as_str()).unwrap_or("?");
                    self.result.add_error("TENSORS",
                        &format!("Tensor '{}' fuera de límites", name), true);
                }
            }
        }
        
        if errors == 0 {
            self.log(&format!("✓ {} tensores validados", tensors.len()));
        } else if errors > 5 {
            self.result.add_error("TENSORS", &format!("... y {} errores más", errors - 5), true);
        }
    }
    
    fn print_summary(&self) {
        println!("\n{}", "=".repeat(72));
        println!("RESUMEN HNF");
        println!("{}", "=".repeat(72));
        
        if self.result.is_valid() {
            println!("\n  ✓ VÁLIDO");
        } else {
            println!("\n  ✗ INVÁLIDO");
        }
        
        println!("    Errores fatales: {}", self.result.fatal_count());
        println!("    Advertencias:    {}", self.result.warn_count());
        
        if self.result.fatal_count() > 0 {
            println!("\n  Errores:");
            for err in &self.result.errors {
                if err.fatal {
                    println!("    • {}", err);
                }
            }
        }
        
        if self.result.warn_count() > 0 {
            println!("\n  Advertencias:");
            for err in &self.result.errors {
                if !err.fatal {
                    println!("    • {}", err);
                }
            }
        }
    }
}

// ============================================================================
// CLI
// ============================================================================

#[derive(Parser, Debug)]
#[command(name = "helios-validate")]
#[command(about = "Validador estricto de formatos HELIOS (HNF, HTF)")]
#[command(version = "1.0.0")]
struct Args {
    /// Archivo a validar (.hnf)
    file: PathBuf,
    
    /// Modo verbose
    #[arg(short, long)]
    verbose: bool,
}

fn main() {
    let args = Args::parse();
    
    if !args.file.exists() {
        eprintln!("Error: Archivo no encontrado: {}", args.file.display());
        std::process::exit(1);
    }
    
    // Leer archivo
    let mut file = match File::open(&args.file) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Error abriendo archivo: {}", e);
            std::process::exit(1);
        }
    };
    
    let mut data = Vec::new();
    if let Err(e) = file.read_to_end(&mut data) {
        eprintln!("Error leyendo archivo: {}", e);
        std::process::exit(1);
    }
    
    // Detectar tipo por magic
    if data.len() < 8 {
        eprintln!("Error: Archivo muy pequeño");
        std::process::exit(1);
    }
    
    let magic = &data[0..8];
    
    let result = if magic == HNF_MAGIC {
        let validator = HnfValidator::new(data, args.verbose);
        validator.validate()
    } else {
        eprintln!("Error: Formato no reconocido (magic: {:?})", magic);
        std::process::exit(1);
    };
    
    println!("\n{}", "=".repeat(72));
    if result.is_valid() {
        println!("✓ VALIDACIÓN EXITOSA");
    } else {
        println!("✗ VALIDACIÓN FALLIDA");
    }
    println!("{}\n", "=".repeat(72));
    
    std::process::exit(if result.is_valid() { 0 } else { 1 });
}
