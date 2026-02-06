// src/hqs/mod.rs
// ============================================================================
// HQS - HELIOS Quantization System
// ============================================================================

pub mod common;
pub mod grid_search;
pub mod hq4k;
pub mod hq5k;

// Re-exports
pub use common::*;
pub use grid_search::GridConfig;
pub use hq4k::{quantize_hq4k, quantize_hq4k_fast, dequantize_hq4k, hq4k_size, validate_hq4k};
pub use hq5k::{quantize_hq5k, quantize_hq5k_fast, dequantize_hq5k, hq5k_size, validate_hq5k};

/// Formato de cuantización
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantFormat {
    FP16,
    HQ3K,
    HQ4K,
    HQ5K,
}

impl QuantFormat {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "FP16" | "FLOAT16" => Some(Self::FP16),
            "HQ3K" | "3BIT" => Some(Self::HQ3K),
            "HQ4K" | "4BIT" => Some(Self::HQ4K),
            "HQ5K" | "5BIT" => Some(Self::HQ5K),
            _ => None,
        }
    }
    
    pub fn code(&self) -> u8 {
        match self {
            Self::FP16 => 0x00,
            Self::HQ3K => 0x01,
            Self::HQ4K => 0x02,
            Self::HQ5K => 0x03,
        }
    }
    
    pub fn bits(&self) -> u8 {
        match self {
            Self::FP16 => 16,
            Self::HQ3K => 3,
            Self::HQ4K => 4,
            Self::HQ5K => 5,
        }
    }
    
    pub fn block_size(&self) -> usize {
        match self {
            Self::FP16 => 2,  // 2 bytes per element
            Self::HQ3K => HQ3K_BLOCK_SIZE,
            Self::HQ4K => HQ4K_BLOCK_SIZE,
            Self::HQ5K => HQ5K_BLOCK_SIZE,
        }
    }
    
    /// Calcula bytes necesarios para N elementos
    pub fn size_for(&self, numel: usize) -> usize {
        match self {
            Self::FP16 => numel * 2,
            Self::HQ3K => {
                let num_blocks = (numel + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
                num_blocks * HQ3K_BLOCK_SIZE
            }
            Self::HQ4K => hq4k_size(numel),
            Self::HQ5K => hq5k_size(numel),
        }
    }
}

impl std::fmt::Display for QuantFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FP16 => write!(f, "FP16"),
            Self::HQ3K => write!(f, "HQ3K"),
            Self::HQ4K => write!(f, "HQ4K"),
            Self::HQ5K => write!(f, "HQ5K"),
        }
    }
}

/// Cuantiza datos según el formato especificado
pub fn quantize(data: &[f32], format: QuantFormat, use_mse: bool) -> Vec<u8> {
    match format {
        QuantFormat::FP16 => {
            // Convertir a f16
            data.iter()
                .flat_map(|&x| half::f16::from_f32(x).to_le_bytes())
                .collect()
        }
        QuantFormat::HQ3K => {
            // TODO: Implementar HQ3K
            unimplemented!("HQ3K not yet implemented")
        }
        QuantFormat::HQ4K => {
            if use_mse {
                quantize_hq4k(data)
            } else {
                quantize_hq4k_fast(data)
            }
        }
        QuantFormat::HQ5K => {
            if use_mse {
                quantize_hq5k(data)
            } else {
                quantize_hq5k_fast(data)
            }
        }
    }
}

/// Dequantiza datos según el formato
pub fn dequantize(data: &[u8], format: QuantFormat, numel: usize) -> Vec<f32> {
    match format {
        QuantFormat::FP16 => {
            data.chunks_exact(2)
                .map(|chunk| {
                    let bytes: [u8; 2] = [chunk[0], chunk[1]];
                    half::f16::from_le_bytes(bytes).to_f32()
                })
                .take(numel)
                .collect()
        }
        QuantFormat::HQ3K => {
            unimplemented!("HQ3K dequantize not yet implemented")
        }
        QuantFormat::HQ4K => dequantize_hq4k(data, numel),
        QuantFormat::HQ5K => dequantize_hq5k(data, numel),
    }
}
