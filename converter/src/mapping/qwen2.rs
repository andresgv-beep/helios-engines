// src/mapping/qwen2.rs
// ============================================================================
// QWEN2 MAPPER - Mapea tensores Qwen2/Qwen2.5 a nombres canónicos HELIOS
// ============================================================================
//
// Soporta: Qwen2, Qwen2.5, Qwen2-Instruct, Qwen2.5-Coder, etc.
// Todos usan la misma arquitectura de tensores.
//
// v9.0.5: Añade soporte para rope_scaling (linear, dynamic, yarn)
//
// ============================================================================

use regex::Regex;
use serde_json::{json, Value};

use super::traits::ModelMapper;
use super::types::{TensorMapping, QuantHint, TensorCategory};

// ============================================================================
// CONFIG
// ============================================================================

#[derive(Debug, Clone)]
pub struct RopeScaling {
    pub scaling_type: String,  // "linear", "dynamic", "yarn", etc.
    pub factor: f64,
}

#[derive(Debug, Clone)]
pub struct Qwen2Config {
    pub num_hidden_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub tie_word_embeddings: bool,
    pub attention_bias: bool,
    // v9.0.5: rope_scaling support
    pub rope_scaling: Option<RopeScaling>,
}

impl Qwen2Config {
    pub fn from_json(config: &Value) -> Self {
        // Parse rope_scaling if present
        let rope_scaling = config.get("rope_scaling")
            .and_then(|rs| {
                if rs.is_null() {
                    return None;
                }
                let scaling_type = rs.get("type")
                    .and_then(|t| t.as_str())
                    .unwrap_or("linear")
                    .to_string();
                let factor = rs.get("factor")
                    .and_then(|f| f.as_f64())
                    .unwrap_or(1.0);
                
                if factor != 1.0 {
                    Some(RopeScaling { scaling_type, factor })
                } else {
                    None
                }
            });
        
        Self {
            num_hidden_layers: config["num_hidden_layers"].as_u64().unwrap_or(32) as usize,
            hidden_size: config["hidden_size"].as_u64().unwrap_or(4096) as usize,
            intermediate_size: config["intermediate_size"].as_u64().unwrap_or(11008) as usize,
            num_attention_heads: config["num_attention_heads"].as_u64().unwrap_or(32) as usize,
            num_key_value_heads: config["num_key_value_heads"].as_u64().unwrap_or(32) as usize,
            vocab_size: config["vocab_size"].as_u64().unwrap_or(151936) as usize,
            max_position_embeddings: config["max_position_embeddings"].as_u64().unwrap_or(32768) as usize,
            rope_theta: config["rope_theta"].as_f64().unwrap_or(10000.0),
            rms_norm_eps: config["rms_norm_eps"].as_f64().unwrap_or(1e-6),
            tie_word_embeddings: config["tie_word_embeddings"].as_bool().unwrap_or(false),
            attention_bias: config["attention_bias"].as_bool().unwrap_or(true),
            rope_scaling,
        }
    }
}

// ============================================================================
// MAPPER
// ============================================================================

pub struct Qwen2Mapper {
    config: Qwen2Config,
    // Compiled regexes
    re_embed: Regex,
    re_lm_head: Regex,
    re_final_norm: Regex,
    re_attn_weight: Regex,
    re_attn_bias: Regex,
    re_mlp_weight: Regex,
    re_input_norm: Regex,
    re_post_attn_norm: Regex,
}

impl Qwen2Mapper {
    pub fn new(config: Qwen2Config) -> Self {
        Self {
            config,
            re_embed: Regex::new(r"^model\.embed_tokens\.weight$").unwrap(),
            re_lm_head: Regex::new(r"^lm_head\.weight$").unwrap(),
            re_final_norm: Regex::new(r"^model\.norm\.weight$").unwrap(),
            re_attn_weight: Regex::new(r"^model\.layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.weight$").unwrap(),
            re_attn_bias: Regex::new(r"^model\.layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.bias$").unwrap(),
            re_mlp_weight: Regex::new(r"^model\.layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight$").unwrap(),
            re_input_norm: Regex::new(r"^model\.layers\.(\d+)\.input_layernorm\.weight$").unwrap(),
            re_post_attn_norm: Regex::new(r"^model\.layers\.(\d+)\.post_attention_layernorm\.weight$").unwrap(),
        }
    }
    
    pub fn from_json(config: &Value) -> Self {
        Self::new(Qwen2Config::from_json(config))
    }
}

impl ModelMapper for Qwen2Mapper {
    fn name(&self) -> &str {
        "qwen2"
    }
    
    fn map_tensor(&self, name: &str) -> Option<TensorMapping> {
        // Ignorar tensores internos
        if self.should_ignore(name) {
            return None;
        }
        
        // ═══════════════════════════════════════════════════════════════
        // EMBEDDINGS (FP16)
        // ═══════════════════════════════════════════════════════════════
        
        if self.re_embed.is_match(name) {
            return Some(TensorMapping::new(
                "token_embedding.weight",
                QuantHint::FP16,
                TensorCategory::Embedding,
            ));
        }
        
        if self.re_lm_head.is_match(name) {
            return Some(TensorMapping::new(
                "lm_head.weight",
                QuantHint::FP16,
                TensorCategory::LMHead,
            ));
        }
        
        if self.re_final_norm.is_match(name) {
            return Some(TensorMapping::new(
                "final_norm.weight",
                QuantHint::FP16,
                TensorCategory::Norm,
            ));
        }
        
        // ═══════════════════════════════════════════════════════════════
        // ATTENTION WEIGHTS (HQ5K - alta precisión)
        // ═══════════════════════════════════════════════════════════════
        
        if let Some(caps) = self.re_attn_weight.captures(name) {
            let layer: usize = caps[1].parse().ok()?;
            let proj = &caps[2];
            return Some(TensorMapping::new(
                format!("layer{}.attn.{}_proj.weight", layer, proj),
                QuantHint::HQ5K,
                TensorCategory::Attention,
            ).with_layer(layer));
        }
        
        // ═══════════════════════════════════════════════════════════════
        // ATTENTION BIASES (FP16)
        // ═══════════════════════════════════════════════════════════════
        
        if let Some(caps) = self.re_attn_bias.captures(name) {
            let layer: usize = caps[1].parse().ok()?;
            let proj = &caps[2];
            return Some(TensorMapping::new(
                format!("layer{}.attn.{}_proj.bias", layer, proj),
                QuantHint::FP16,
                TensorCategory::Attention,
            ).with_layer(layer));
        }
        
        // ═══════════════════════════════════════════════════════════════
        // MLP WEIGHTS (HQ4K - buena compresión)
        // ═══════════════════════════════════════════════════════════════
        
        if let Some(caps) = self.re_mlp_weight.captures(name) {
            let layer: usize = caps[1].parse().ok()?;
            let proj = &caps[2];
            return Some(TensorMapping::new(
                format!("layer{}.mlp.{}.weight", layer, proj),
                QuantHint::HQ4K,
                TensorCategory::MLP,
            ).with_layer(layer));
        }
        
        // ═══════════════════════════════════════════════════════════════
        // LAYER NORMS (FP16)
        // ═══════════════════════════════════════════════════════════════
        
        if let Some(caps) = self.re_input_norm.captures(name) {
            let layer: usize = caps[1].parse().ok()?;
            return Some(TensorMapping::new(
                format!("layer{}.ln_attn_in.weight", layer),
                QuantHint::FP16,
                TensorCategory::Norm,
            ).with_layer(layer));
        }
        
        if let Some(caps) = self.re_post_attn_norm.captures(name) {
            let layer: usize = caps[1].parse().ok()?;
            return Some(TensorMapping::new(
                format!("layer{}.ln_attn_out.weight", layer),
                QuantHint::FP16,
                TensorCategory::Norm,
            ).with_layer(layer));
        }
        
        // No mapeado
        None
    }
    
    fn execution_hints(&self) -> Value {
        let c = &self.config;
        
        let attention_type = if c.num_key_value_heads == c.num_attention_heads {
            "mha"
        } else if c.num_key_value_heads == 1 {
            "mqa"
        } else {
            "gqa"
        };
        
        let head_dim = c.hidden_size / c.num_attention_heads;
        
        // Determinar rope_type basado en rope_scaling
        let rope_type = match &c.rope_scaling {
            Some(rs) => rs.scaling_type.as_str(),
            None => "default",
        };
        
        let mut hints = json!({
            // IDENTIFICACIÓN (OBLIGATORIO)
            "arch": "qwen2",
            "dtype": "bf16",
            
            // DIMENSIONES (OBLIGATORIO)
            "num_hidden_layers": c.num_hidden_layers,
            "hidden_size": c.hidden_size,
            "intermediate_size": c.intermediate_size,
            "vocab_size": c.vocab_size,
            
            // ATTENTION (OBLIGATORIO)
            "num_attention_heads": c.num_attention_heads,
            "num_key_value_heads": c.num_key_value_heads,
            "head_dim": head_dim,
            "attention_type": attention_type,
            "attention_bias": c.attention_bias,
            "qkv_layout": "separate",
            "use_qk_norm": false,
            "parallel_attention": false,
            "kv_layout": "BHSD",
            
            // MLP (OBLIGATORIO)
            "mlp_type": "swiglu",
            "mlp_activation": "silu",
            "mlp_bias": false,
            
            // NORMALIZATION (OBLIGATORIO)
            "norm_type": "rmsnorm",
            "norm_bias": false,
            "rms_norm_eps": c.rms_norm_eps,
            "pre_norm": true,
            "final_norm": true,
            
            // RoPE (OBLIGATORIO)
            "rope_type": rope_type,
            "rope_theta": c.rope_theta,
            "rope_dim": head_dim,
            "rope_partial": false,
            "rope_interleaved": false,
            
            // EMBEDDINGS (OBLIGATORIO)
            "tie_word_embeddings": c.tie_word_embeddings,
            "embedding_bias": false,
            "lm_head_bias": false,
            
            // CONTEXT
            "max_position_embeddings": c.max_position_embeddings,
            
            // MoE
            "moe_enabled": false,
            
            // INFERENCE CAPABILITIES
            "supports_flash_attention": true,
            "supports_paged_attention": true,
            "supports_sdpa": true
        });
        
        // v9.0.5: Añadir rope_scaling si está presente
        if let Some(rs) = &c.rope_scaling {
            hints["rope_scaling"] = json!({
                "type": rs.scaling_type,
                "factor": rs.factor
            });
        }
        
        hints
    }
    
    fn num_layers(&self) -> usize {
        self.config.num_hidden_layers
    }
    
    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }
    
    fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }
}
