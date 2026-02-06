// src/main.rs
// ============================================================================
// HELIOS-CONVERT CLI
// ============================================================================
//
// Uso simple:
//   helios-convert ./Qwen2-7B -o qwen.hnf
//
// Uso multimodal:
//   helios-convert \
//       --text ./Qwen2-7B \
//       --vision ./SigLIP-base \
//       --code ./Qwen2.5-Coder-7B \
//       --cortex ./Phi-4-mini \
//       -o helios_core.hnf
//
// ============================================================================

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use clap::Parser;

use helios_convert::{
    hqs::QuantFormat,
    hnf::HnfWriter,
    mapping::{BlockType, create_mapper, ModelMapper},
    builder::{process_model, write_combined_hints, BuildStats},
    htf::{self, DomainType},
};

#[derive(Parser, Debug)]
#[command(name = "helios-convert")]
#[command(about = "Convert HuggingFace models to HNFv9 format")]
#[command(version = "0.2.1")]
struct Args {
    /// Input model (shorthand for --text)
    #[arg(value_name = "MODEL")]
    model: Option<PathBuf>,
    
    /// Text/LLM model → block 0x0
    #[arg(long)]
    text: Option<PathBuf>,
    
    /// Vision encoder → block 0x1
    #[arg(long)]
    vision: Option<PathBuf>,
    
    /// Audio encoder → block 0x2
    #[arg(long)]
    audio: Option<PathBuf>,
    
    /// Cortex/reasoning model → block 0x7
    #[arg(long)]
    cortex: Option<PathBuf>,
    
    /// Code model → block 0x8
    #[arg(long)]
    code: Option<PathBuf>,
    
    /// Output HNF file
    #[arg(short, long, required = true)]
    output: PathBuf,
    
    /// Default quantization format
    #[arg(short, long, default_value = "HQ5K")]
    quant: String,
    
    /// Skip MSE optimization (faster, lower quality)
    #[arg(long)]
    fast: bool,
    
    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let start = Instant::now();
    
    // Parse quant format
    let default_quant = QuantFormat::from_str(&args.quant)
        .ok_or_else(|| anyhow::anyhow!("Invalid quant format: {}", args.quant))?;
    
    let use_mse = !args.fast;
    
    // Resolver modelo de texto (positional o --text)
    let text_model = args.text.or(args.model);
    
    // Validar que hay al menos un modelo
    if text_model.is_none() 
        && args.vision.is_none() 
        && args.audio.is_none() 
        && args.cortex.is_none() 
        && args.code.is_none() 
    {
        anyhow::bail!("No model specified. Use positional argument or --text/--vision/--audio/--cortex/--code");
    }
    
    println!("═══════════════════════════════════════════════════════════════");
    println!("  HELIOS CONVERTER v0.2.1 - HQS v6 Nuclear + Multi-Tokenizer");
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Default quant: {}", default_quant);
    println!("  MSE search:    {}", if use_mse { "ON" } else { "OFF (fast)" });
    println!("  Output:        {}", args.output.display());
    println!("═══════════════════════════════════════════════════════════════");
    
    // Crear writer
    let mut writer = HnfWriter::create(&args.output)?;
    
    // Recolectar mappers para hints combinados
    let mut mappers: Vec<(Box<dyn ModelMapper>, BlockType)> = Vec::new();
    let mut total_stats = BuildStats::default();
    
    // ══════════════════════════════════════════════════════════════════════
    // PROCESAR CADA MODELO
    // ══════════════════════════════════════════════════════════════════════
    
    if let Some(path) = &text_model {
        println!("\n[TEXT] {} → block 0x0", path.display());
        let stats = process_model(path, BlockType::TextModel, &mut writer, default_quant, use_mse, args.verbose)?;
        println!("  ✓ {} tensors (FP16:{}, HQ5K:{}, HQ4K:{})", 
            stats.total_tensors(), stats.fp16_count, stats.hq5k_count, stats.hq4k_count);
        
        let mapper = create_mapper(path)?;
        mappers.push((mapper, BlockType::TextModel));
        merge_stats(&mut total_stats, &stats);
    }
    
    if let Some(path) = &args.vision {
        println!("\n[VISION] {} → block 0x1", path.display());
        let stats = process_model(path, BlockType::Vision, &mut writer, default_quant, use_mse, args.verbose)?;
        println!("  ✓ {} tensors (FP16:{}, HQ5K:{}, HQ4K:{})", 
            stats.total_tensors(), stats.fp16_count, stats.hq5k_count, stats.hq4k_count);
        
        let mapper = create_mapper(path)?;
        mappers.push((mapper, BlockType::Vision));
        merge_stats(&mut total_stats, &stats);
    }
    
    if let Some(path) = &args.audio {
        println!("\n[AUDIO] {} → block 0x2", path.display());
        let stats = process_model(path, BlockType::Audio, &mut writer, default_quant, use_mse, args.verbose)?;
        println!("  ✓ {} tensors (FP16:{}, HQ5K:{}, HQ4K:{})", 
            stats.total_tensors(), stats.fp16_count, stats.hq5k_count, stats.hq4k_count);
        
        let mapper = create_mapper(path)?;
        mappers.push((mapper, BlockType::Audio));
        merge_stats(&mut total_stats, &stats);
    }
    
    if let Some(path) = &args.cortex {
        println!("\n[CORTEX] {} → block 0x7", path.display());
        let stats = process_model(path, BlockType::Cortex, &mut writer, default_quant, use_mse, args.verbose)?;
        println!("  ✓ {} tensors (FP16:{}, HQ5K:{}, HQ4K:{})", 
            stats.total_tensors(), stats.fp16_count, stats.hq5k_count, stats.hq4k_count);
        
        let mapper = create_mapper(path)?;
        mappers.push((mapper, BlockType::Cortex));
        merge_stats(&mut total_stats, &stats);
    }
    
    if let Some(path) = &args.code {
        println!("\n[CODE] {} → block 0x8", path.display());
        let stats = process_model(path, BlockType::CodeExec, &mut writer, default_quant, use_mse, args.verbose)?;
        println!("  ✓ {} tensors (FP16:{}, HQ5K:{}, HQ4K:{})", 
            stats.total_tensors(), stats.fp16_count, stats.hq5k_count, stats.hq4k_count);
        
        let mapper = create_mapper(path)?;
        mappers.push((mapper, BlockType::CodeExec));
        merge_stats(&mut total_stats, &stats);
    }
    
    // ══════════════════════════════════════════════════════════════════════
    // EXECUTION HINTS
    // ══════════════════════════════════════════════════════════════════════
    
    println!("\n[HINTS] Writing execution hints...");
    let mapper_refs: Vec<(&dyn ModelMapper, BlockType)> = mappers
        .iter()
        .map(|(m, b)| (m.as_ref(), *b))
        .collect();
    write_combined_hints(&mut writer, &mapper_refs)?;
    println!("  ✓ Done");
    
    // ══════════════════════════════════════════════════════════════════════
    // TOKENIZER (MULTI-DOMAIN)
    // ══════════════════════════════════════════════════════════════════════
    
    println!("\n[TOKENIZER] Writing tokenizers (multi-domain)...");
    
    // Construir lista de fuentes de tokenizer
    let mut tok_sources: Vec<(&std::path::Path, DomainType, bool)> = Vec::new();
    
    // TEXT es siempre primario si existe
    if let Some(path) = text_model.as_ref() {
        tok_sources.push((path.as_path(), DomainType::Text, true));
    }
    
    // CODE como dominio secundario
    if let Some(path) = args.code.as_ref() {
        tok_sources.push((path.as_path(), DomainType::Code, false));
    }
    
    // CORTEX como dominio secundario (usa TEXT domain type ya que es LLM)
    if let Some(path) = args.cortex.as_ref() {
        // Cortex es otro LLM, podría compartir tokenizer con text o tener el suyo
        // Por ahora lo añadimos como TEXT secundario
        tok_sources.push((path.as_path(), DomainType::Text, false));
    }
    
    // AUDIO si tiene tokenizer
    if let Some(path) = args.audio.as_ref() {
        tok_sources.push((path.as_path(), DomainType::Audio, false));
    }
    
    // Construir HTF multi-domain
    if !tok_sources.is_empty() {
        let htf_bytes = htf::build_htf_multi(&tok_sources)?;
        writer.write_tokenizer(&htf_bytes)?;
        println!("  ✓ {} bytes ({} domains)", htf_bytes.len(), tok_sources.len());
    } else {
        println!("  ⚠ No tokenizers found");
    }
    
    // ══════════════════════════════════════════════════════════════════════
    // FINALIZE
    // ══════════════════════════════════════════════════════════════════════
    
    println!("\n[FINALIZE] Writing manifest...");
    let manifest = serde_json::json!({
        "format": "HNFv9",
        "version": "9.0.1",
        "generator": "helios-convert 0.2.1",
        "quantization": {
            "default": args.quant,
            "hqs_version": "v6-nuclear",
            "mse_search": use_mse,
        },
        "stats": {
            "total_tensors": total_stats.total_tensors(),
            "fp16": total_stats.fp16_count,
            "hq5k": total_stats.hq5k_count,
            "hq4k": total_stats.hq4k_count,
            "skipped": total_stats.skipped_count,
        },
        "tokenizer": {
            "multi_domain": true,
            "domains": tok_sources.len(),
        }
    });
    writer.finalize(manifest)?;
    
    // ══════════════════════════════════════════════════════════════════════
    // SUMMARY
    // ══════════════════════════════════════════════════════════════════════
    
    let elapsed = start.elapsed();
    let file_size = std::fs::metadata(&args.output)?.len();
    
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  CONVERSION COMPLETE");
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Time:       {:.1}s", elapsed.as_secs_f64());
    println!("  Size:       {:.1} MB", file_size as f64 / 1024.0 / 1024.0);
    println!("  Tensors:    {} (FP16:{}, HQ5K:{}, HQ4K:{})", 
        total_stats.total_tensors(),
        total_stats.fp16_count,
        total_stats.hq5k_count,
        total_stats.hq4k_count);
    println!("  Skipped:    {}", total_stats.skipped_count);
    println!("  Tokenizers: {} domains", tok_sources.len());
    println!("  Output:     {}", args.output.display());
    println!("═══════════════════════════════════════════════════════════════");
    
    Ok(())
}

fn merge_stats(total: &mut BuildStats, part: &BuildStats) {
    total.fp16_count += part.fp16_count;
    total.hq5k_count += part.hq5k_count;
    total.hq4k_count += part.hq4k_count;
    total.skipped_count += part.skipped_count;
    total.total_bytes += part.total_bytes;
}
