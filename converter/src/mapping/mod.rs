// src/mapping/mod.rs
// ============================================================================
// MAPPING - Sistema de mapeo de tensores HELIOS
// ============================================================================

pub mod types;
pub mod traits;
pub mod factory;
pub mod qwen2;
pub mod llama;
pub mod clip;
pub mod phi;  // AÑADIDO

// Re-exports
pub use types::{BlockType, QuantHint, TensorCategory, TensorMapping};
pub use traits::ModelMapper;
pub use factory::{create_mapper, detect_architecture, load_config};
