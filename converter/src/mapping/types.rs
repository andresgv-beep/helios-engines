// src/mapping/types.rs
// ============================================================================
// MAPPING TYPES - Tipos básicos para el sistema de mapeo
// ============================================================================

/// Bloque HNF destino
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BlockType {
    TextModel = 0x0,
    Vision = 0x1,
    Audio = 0x2,
    Video = 0x3,
    Spatial3D = 0x4,
    Personality = 0x5,
    Memory = 0x6,
    Cortex = 0x7,
    CodeExec = 0x8,
    Tokenizer = 0x9,      // HTF tokenizer
    ExecutionHints = 0xA,
    ExpertRouter = 0xB,
    Tools = 0xC,          // Movido aquí
}

impl BlockType {
    pub fn as_usize(&self) -> usize {
        *self as usize
    }
    
    pub fn name(&self) -> &'static str {
        match self {
            Self::TextModel => "text_model",
            Self::Vision => "vision",
            Self::Audio => "audio",
            Self::Video => "video",
            Self::Spatial3D => "spatial_3d",
            Self::Personality => "personality",
            Self::Memory => "memory",
            Self::Cortex => "cortex",
            Self::CodeExec => "code_exec",
            Self::Tokenizer => "tokenizer",
            Self::ExecutionHints => "execution_hints",
            Self::ExpertRouter => "expert_router",
            Self::Tools => "tools",
        }
    }
}

/// Hint de cuantización - el mapper sugiere, el builder decide
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantHint {
    /// FP16 - embeddings, norms, biases (sin pérdida)
    FP16,
    /// HQ5K - atención, routers (alta precisión)
    HQ5K,
    /// HQ4K - MLP, expertos (buena compresión)
    HQ4K,
    /// Usar el default del CLI
    Default,
}

impl QuantHint {
    pub fn resolve(&self, default: crate::hqs::QuantFormat) -> crate::hqs::QuantFormat {
        match self {
            Self::FP16 => crate::hqs::QuantFormat::FP16,
            Self::HQ5K => crate::hqs::QuantFormat::HQ5K,
            Self::HQ4K => crate::hqs::QuantFormat::HQ4K,
            Self::Default => default,
        }
    }
}

/// Categoría del tensor (para hints y estadísticas)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorCategory {
    Embedding,
    Attention,
    MLP,
    Norm,
    LMHead,
    MoERouter,
    MoEExpert,
    VisionPatch,
    VisionProjector,
    AudioMel,
    Other,
}

/// Resultado del mapeo de un tensor
#[derive(Debug, Clone)]
pub struct TensorMapping {
    /// Nombre canónico HELIOS (ej: "layer0.attn.q_proj.weight")
    pub canonical_name: String,
    /// Sugerencia de cuantización
    pub quant_hint: QuantHint,
    /// Categoría del tensor
    pub category: TensorCategory,
    /// Índice de capa (si aplica)
    pub layer_idx: Option<usize>,
    /// Índice de experto MoE (si aplica)
    pub expert_idx: Option<usize>,
}

impl TensorMapping {
    pub fn new(canonical_name: impl Into<String>, quant_hint: QuantHint, category: TensorCategory) -> Self {
        Self {
            canonical_name: canonical_name.into(),
            quant_hint,
            category,
            layer_idx: None,
            expert_idx: None,
        }
    }
    
    pub fn with_layer(mut self, layer: usize) -> Self {
        self.layer_idx = Some(layer);
        self
    }
    
    pub fn with_expert(mut self, expert: usize) -> Self {
        self.expert_idx = Some(expert);
        self
    }
}
