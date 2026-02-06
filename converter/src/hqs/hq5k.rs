// src/hqs/hq5k.rs
// ============================================================================
// HQ5K v6 - NUCLEAR: grupos de 8 elementos
// ============================================================================

use rayon::prelude::*;
use crate::hqs::common::*;
use crate::hqs::grid_search::*;

const Q_MAX: f32 = 31.0;

fn quantize_superblock(block: &[f32; SUPER_BLOCK_SIZE], use_mse: bool) -> Vec<u8> {
    let config = GridConfig::hq5k();
    
    let group_params = if use_mse {
        optimize_superblock(block, &config)
    } else {
        fast_superblock(block)
    };
    
    let mut q_indices = [0u8; SUPER_BLOCK_SIZE];
    
    for g in 0..NUM_GROUPS {
        let gp = &group_params[g];
        let start = g * GROUP_SIZE;
        
        for i in 0..GROUP_SIZE {
            let val = block[start + i];
            let q = ((val - gp.min) / gp.scale * Q_MAX).round().clamp(0.0, Q_MAX) as u8;
            q_indices[start + i] = q;
        }
    }
    
    let mut output = vec![0u8; HQ5K_BLOCK_SIZE];
    
    let header = encode_header(&group_params);
    output[..HEADER_SIZE].copy_from_slice(&header);
    
    // Pack 5-bit LSB-first: 8 valores = 40 bits = 5 bytes
    for chunk_idx in 0..(SUPER_BLOCK_SIZE / 8) {
        let base = chunk_idx * 8;
        
        let mut bits: u64 = 0;
        for k in 0..8 {
            bits |= (q_indices[base + k] as u64 & 0x1F) << (k * 5);
        }
        
        let byte_idx = HEADER_SIZE + chunk_idx * 5;
        for k in 0..5 {
            output[byte_idx + k] = ((bits >> (k * 8)) & 0xFF) as u8;
        }
    }
    
    output
}

pub fn quantize_hq5k(data: &[f32]) -> Vec<u8> {
    quantize_hq5k_internal(data, true)
}

pub fn quantize_hq5k_fast(data: &[f32]) -> Vec<u8> {
    quantize_hq5k_internal(data, false)
}

fn quantize_hq5k_internal(data: &[f32], use_mse: bool) -> Vec<u8> {
    let padded = pad_to_superblock(data);
    let num_blocks = padded.len() / SUPER_BLOCK_SIZE;
    
    if num_blocks == 0 {
        return Vec::new();
    }
    
    let results: Vec<Vec<u8>> = (0..num_blocks)
        .into_par_iter()
        .map(|b| {
            let start = b * SUPER_BLOCK_SIZE;
            let mut block = [0.0f32; SUPER_BLOCK_SIZE];
            for i in 0..SUPER_BLOCK_SIZE {
                let val = padded[start + i];
                block[i] = if val.is_finite() { val } else { 0.0 };
            }
            quantize_superblock(&block, use_mse)
        })
        .collect();
    
    let mut output = Vec::with_capacity(num_blocks * HQ5K_BLOCK_SIZE);
    for block_data in results {
        output.extend(block_data);
    }
    
    output
}

pub fn dequantize_hq5k(data: &[u8], numel: usize) -> Vec<f32> {
    if data.is_empty() {
        return vec![0.0; numel];
    }
    
    let num_blocks = data.len() / HQ5K_BLOCK_SIZE;
    let mut output = Vec::with_capacity(num_blocks * SUPER_BLOCK_SIZE);
    
    for b in 0..num_blocks {
        let block_start = b * HQ5K_BLOCK_SIZE;
        let header: [u8; HEADER_SIZE] = data[block_start..block_start + HEADER_SIZE]
            .try_into()
            .unwrap();
        
        let group_params = decode_header(&header);
        
        let payload_start = block_start + HEADER_SIZE;
        let mut q_indices = [0u8; SUPER_BLOCK_SIZE];
        
        for chunk_idx in 0..(SUPER_BLOCK_SIZE / 8) {
            let byte_idx = payload_start + chunk_idx * 5;
            
            let mut bits: u64 = 0;
            for k in 0..5 {
                bits |= (data[byte_idx + k] as u64) << (k * 8);
            }
            
            for k in 0..8 {
                q_indices[chunk_idx * 8 + k] = ((bits >> (k * 5)) & 0x1F) as u8;
            }
        }
        
        for g in 0..NUM_GROUPS {
            let gp = &group_params[g];
            let start = g * GROUP_SIZE;
            
            for i in 0..GROUP_SIZE {
                let q = q_indices[start + i] as f32;
                let val = gp.min + q / Q_MAX * gp.scale;
                output.push(val);
            }
        }
    }
    
    output.truncate(numel);
    output
}

pub fn hq5k_size(numel: usize) -> usize {
    let num_blocks = (numel + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    num_blocks * HQ5K_BLOCK_SIZE
}

pub fn validate_hq5k(data: &[u8], numel: usize) -> Result<(), String> {
    let expected_size = hq5k_size(numel);
    if data.len() != expected_size {
        return Err(format!("Size mismatch: expected {}, got {}", expected_size, data.len()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    
    #[test]
    fn test_roundtrip() {
        let mut rng = rand::thread_rng();
        let original: Vec<f32> = (0..1024).map(|_| rng.gen_range(-2.0..2.0)).collect();
        
        let quantized = quantize_hq5k(&original);
        let recovered = dequantize_hq5k(&quantized, original.len());
        
        assert_eq!(recovered.len(), original.len());
        
        let orig_mean: f32 = original.iter().sum::<f32>() / original.len() as f32;
        let rec_mean: f32 = recovered.iter().sum::<f32>() / recovered.len() as f32;
        
        let mut cov = 0.0f32;
        let mut var_orig = 0.0f32;
        let mut var_rec = 0.0f32;
        
        for (o, r) in original.iter().zip(recovered.iter()) {
            let do_ = o - orig_mean;
            let dr = r - rec_mean;
            cov += do_ * dr;
            var_orig += do_ * do_;
            var_rec += dr * dr;
        }
        
        let correlation = cov / (var_orig.sqrt() * var_rec.sqrt());
        println!("HQ5K Correlation: {:.4}", correlation);
        assert!(correlation > 0.999, "Correlation too low: {}", correlation);
    }
    
    #[test]
    fn test_error_rate() {
        let mut rng = rand::thread_rng();
        let original: Vec<f32> = (0..10240).map(|_| rng.gen_range(-2.0..2.0)).collect();
        
        let quantized = quantize_hq5k(&original);
        let recovered = dequantize_hq5k(&quantized, original.len());
        
        let orig_std = {
            let mean: f32 = original.iter().sum::<f32>() / original.len() as f32;
            let var: f32 = original.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / original.len() as f32;
            var.sqrt()
        };
        
        let mse: f32 = original.iter().zip(recovered.iter())
            .map(|(o, r)| (o - r).powi(2))
            .sum::<f32>() / original.len() as f32;
        
        let relative_error = mse.sqrt() / orig_std;
        println!("HQ5K Relative error: {:.2}%", relative_error * 100.0);
        
        // Target: <3% con grupos de 8
        assert!(relative_error < 0.03, "Error {:.2}% exceeds 3%", relative_error * 100.0);
    }
}
