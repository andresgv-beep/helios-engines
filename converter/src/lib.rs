// src/lib.rs
// ============================================================================
// HELIOS-CONVERT - Conversor de Safetensors a HNFv9
// ============================================================================

#![recursion_limit = "256"]

pub mod hnf;
pub mod htf;
pub mod hqs;
pub mod mapping;
pub mod safetensor;
pub mod hints;
pub mod builder;
pub mod dictionary;

// Re-exports principales
pub use hnf::HnfWriter;
pub use hqs::{QuantFormat, quantize, dequantize};
pub use safetensor::SafetensorReader;
pub use mapping::{ModelMapper, BlockType, QuantHint, TensorMapping, create_mapper};
pub use builder::{process_model, write_combined_hints, BuildStats};
