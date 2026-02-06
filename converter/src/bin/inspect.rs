// src/bin/inspect.rs
// ============================================================================
// HNF INSPECTOR - Inspecciona estructura de archivos HNFv9
// ============================================================================
//
// Uso: helios-inspect archivo.hnf
//
// ============================================================================

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

use anyhow::{Result, Context};
use clap::Parser;

#[derive(Parser)]
#[command(name = "helios-inspect")]
#[command(about = "Inspect HNFv9 file structure")]
struct Args {
    /// HNF file to inspect
    file: PathBuf,
    
    /// Show manifest JSON
    #[arg(long)]
    manifest: bool,
    
    /// Show execution hints JSON
    #[arg(long)]
    hints: bool,
}

const BLOCK_NAMES: [&str; 16] = [
    "text_model",       // 0x0
    "vision",           // 0x1
    "audio",            // 0x2
    "video",            // 0x3
    "spatial_3d",       // 0x4
    "personality",      // 0x5
    "memory",           // 0x6
    "cortex",           // 0x7
    "code_exec",        // 0x8
    "tokenizer",        // 0x9 - HTF tokenizer
    "execution_hints",  // 0xA
    "expert_router",    // 0xB
    "tools",            // 0xC
    "reserved_1",       // 0xD
    "reserved_2",       // 0xE
    "reserved_3",       // 0xF
];

const FLAG_NAMES: [(u32, &str); 12] = [
    (0, "HAS_VISION"),
    (1, "HAS_AUDIO"),
    (2, "HAS_VIDEO"),
    (3, "HAS_SPATIAL"),
    (4, "HAS_PERSONALITY"),
    (5, "HAS_MEMORY"),
    (6, "HAS_CORTEX"),
    (7, "HAS_CODE_EXEC"),
    (8, "HAS_TOOLS"),
    (9, "HAS_EXPERT_ROUTER"),
    (10, "IS_MOE"),
    (11, "IS_MULTIMODAL"),
];

#[derive(Debug)]
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

#[derive(Debug)]
struct BlockEntry {
    id: u32,
    block_type: u32,
    offset: u64,
    size: u64,
    checksum: u64,
}

fn format_size(size: u64) -> String {
    if size == 0 {
        "vacío".to_string()
    } else if size < 1024 {
        format!("{} B", size)
    } else if size < 1024 * 1024 {
        format!("{:.1} KB", size as f64 / 1024.0)
    } else if size < 1024 * 1024 * 1024 {
        format!("{:.1} MB", size as f64 / 1024.0 / 1024.0)
    } else {
        format!("{:.2} GB", size as f64 / 1024.0 / 1024.0 / 1024.0)
    }
}

fn make_bar(size: u64, max_size: u64, width: usize) -> String {
    if max_size == 0 || size == 0 {
        "░".repeat(width)
    } else {
        let ratio = (size as f64 / max_size as f64).min(1.0);
        let filled = (ratio * width as f64) as usize;
        // Mínimo 1 bloque si hay datos
        let filled = filled.max(1);
        "█".repeat(filled) + &"░".repeat(width.saturating_sub(filled))
    }
}

/// Crea barra con escala logarítmica para mejor visualización
fn make_bar_log(size: u64, width: usize) -> String {
    if size == 0 {
        return "░".repeat(width);
    }
    
    // Escala logarítmica: 1B -> 0%, 1TB -> 100%
    let log_size = (size as f64).log10();
    let log_max = 12.0; // 10^12 = 1TB
    let ratio = (log_size / log_max).min(1.0).max(0.0);
    let filled = (ratio * width as f64) as usize;
    
    "█".repeat(filled.max(1)) + &"░".repeat(width.saturating_sub(filled.max(1)))
}

fn read_header(f: &mut File) -> Result<HnfHeader> {
    let mut buf = [0u8; 64];
    f.read_exact(&mut buf)?;
    
    Ok(HnfHeader {
        magic: buf[0..8].try_into().unwrap(),
        version_major: u16::from_le_bytes(buf[8..10].try_into().unwrap()),
        version_minor: u16::from_le_bytes(buf[10..12].try_into().unwrap()),
        flags: u32::from_le_bytes(buf[12..16].try_into().unwrap()),
        block_count: u32::from_le_bytes(buf[16..20].try_into().unwrap()),
        header_size: u32::from_le_bytes(buf[20..24].try_into().unwrap()),
        block_table_offset: u64::from_le_bytes(buf[24..32].try_into().unwrap()),
        manifest_offset: u64::from_le_bytes(buf[32..40].try_into().unwrap()),
        manifest_size: u64::from_le_bytes(buf[40..48].try_into().unwrap()),
        file_size: u64::from_le_bytes(buf[48..56].try_into().unwrap()),
        checksum: u32::from_le_bytes(buf[56..60].try_into().unwrap()),
    })
}

fn read_block_table(f: &mut File, offset: u64) -> Result<Vec<BlockEntry>> {
    f.seek(SeekFrom::Start(offset))?;
    
    let mut blocks = Vec::with_capacity(16);
    
    for _ in 0..16 {
        let mut buf = [0u8; 32];
        f.read_exact(&mut buf)?;
        
        blocks.push(BlockEntry {
            id: u32::from_le_bytes(buf[0..4].try_into().unwrap()),
            block_type: u32::from_le_bytes(buf[4..8].try_into().unwrap()),
            offset: u64::from_le_bytes(buf[8..16].try_into().unwrap()),
            size: u64::from_le_bytes(buf[16..24].try_into().unwrap()),
            checksum: u64::from_le_bytes(buf[24..32].try_into().unwrap()),
        });
    }
    
    Ok(blocks)
}

fn main() -> Result<()> {
    let args = Args::parse();
    
    let file_size = std::fs::metadata(&args.file)?.len();
    let mut f = File::open(&args.file)
        .with_context(|| format!("Cannot open {}", args.file.display()))?;
    
    // Leer header
    let header = read_header(&mut f)?;
    
    // Validar magic
    let expected_magic = b"HNFv9\x00\x00\x00";
    let magic_ok = &header.magic == expected_magic;
    
    println!();
    println!("════════════════════════════════════════════════════════════════════════════════");
    println!("  HNFv9 INSPECTOR");
    println!("════════════════════════════════════════════════════════════════════════════════");
    println!("  Archivo:      {}", args.file.display());
    println!("  Tamaño real:  {}", format_size(file_size));
    println!();
    
    // ═══════════════════════════════════════════════════════════════
    // HEADER
    // ═══════════════════════════════════════════════════════════════
    println!("┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│ HEADER (64 bytes)                                                            │");
    println!("├──────────────────────────────────────────────────────────────────────────────┤");
    
    let magic_str: String = header.magic.iter()
        .map(|&b| if b == 0 { '.' } else { b as char })
        .collect();
    let status = if magic_ok { "✓" } else { "✗ INVÁLIDO" };
    
    println!("│  Magic:          {:20} {}                          │", format!("{:?}", magic_str), status);
    println!("│  Versión:        {}.{}                                                        │", header.version_major, header.version_minor);
    println!("│  Block Count:    {:4}                                                        │", header.block_count);
    println!("│  Header Size:    {:4}                                                        │", header.header_size);
    println!("│  File Size:      {:12}                                              │", format_size(header.file_size));
    println!("│  Checksum:       0x{:08X}                                                  │", header.checksum);
    println!("└──────────────────────────────────────────────────────────────────────────────┘");
    println!();
    
    // ═══════════════════════════════════════════════════════════════
    // FLAGS
    // ═══════════════════════════════════════════════════════════════
    println!("┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│ FLAGS                                                                        │");
    println!("├──────────────────────────────────────────────────────────────────────────────┤");
    
    let mut active_flags = Vec::new();
    for (bit, name) in FLAG_NAMES.iter() {
        if header.flags & (1 << bit) != 0 {
            active_flags.push(*name);
        }
    }
    
    if active_flags.is_empty() {
        println!("│  (ningún flag activo)                                                        │");
    } else {
        for flag in &active_flags {
            println!("│  ✓ {:72} │", flag);
        }
    }
    
    println!("└──────────────────────────────────────────────────────────────────────────────┘");
    println!();
    
    // ═══════════════════════════════════════════════════════════════
    // BLOCK TABLE
    // ═══════════════════════════════════════════════════════════════
    let blocks = read_block_table(&mut f, header.block_table_offset)?;
    
    // Encontrar max_size solo de bloques de datos (no hints/tokenizer)
    let max_data_size = blocks.iter()
        .filter(|b| b.id < 10) // Solo bloques de datos
        .map(|b| b.size)
        .max()
        .unwrap_or(1);
    
    println!("┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│ BLOQUES                                                                      │");
    println!("├──────────────────────────────────────────────────────────────────────────────┤");
    
    // Categorías
    let categories = [
        ("MODALIDADES", vec![0, 1, 2, 3, 4]),
        ("IDENTIDAD", vec![5, 6]),
        ("CAPACIDADES", vec![7, 8, 9]),
        ("RUNTIME", vec![10]),
        ("EXPERTS", vec![11]),
    ];
    
    for (cat_name, indices) in &categories {
        println!("│  {}                                                                      │", cat_name);
        println!("│  ────────────────────────────────────────────────────────────────────────  │");
        
        for &idx in indices {
            let b = &blocks[idx];
            let name = BLOCK_NAMES[idx];
            
            // Usar escala logarítmica para mejor visualización
            let bar = make_bar_log(b.size, 25);
            let size_str = format_size(b.size);
            
            let status = if b.size > 0 { "█" } else { "░" };
            
            println!("│  [{}] {:18} {} {:>12}  {}               │", 
                format!("{:X}", idx), name, bar, size_str, status);
        }
        println!("│                                                                              │");
    }
    
    println!("└──────────────────────────────────────────────────────────────────────────────┘");
    println!();
    
    // ═══════════════════════════════════════════════════════════════
    // TOKENIZER & MANIFEST
    // ═══════════════════════════════════════════════════════════════
    
    // Calcular offset del tokenizer
    let last_block_end = blocks.iter()
        .filter(|b| b.size > 0)
        .map(|b| b.offset + b.size)
        .max()
        .unwrap_or(0);
    
    let tok_offset = ((last_block_end + 31) / 32) * 32;
    let tok_size = if header.manifest_offset > tok_offset {
        header.manifest_offset - tok_offset
    } else {
        0
    };
    
    println!("┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│ TOKENIZER & MANIFEST                                                         │");
    println!("├──────────────────────────────────────────────────────────────────────────────┤");
    println!("│  Tokenizer:                                                                  │");
    println!("│    Offset:       0x{:X}                                                      │", tok_offset);
    println!("│    Size:         {:12}                                              │", format_size(tok_size));
    println!("│                                                                              │");
    println!("│  Manifest:                                                                   │");
    println!("│    Offset:       0x{:X}                                                   │", header.manifest_offset);
    println!("│    Size:         {:12}                                              │", format_size(header.manifest_size));
    println!("└──────────────────────────────────────────────────────────────────────────────┘");
    println!();
    
    // ═══════════════════════════════════════════════════════════════
    // MAPA DE ARCHIVO
    // ═══════════════════════════════════════════════════════════════
    println!("┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│ MAPA DE ARCHIVO                                                              │");
    println!("├──────────────────────────────────────────────────────────────────────────────┤");
    
    let mut sections: Vec<(&str, u64, u64)> = vec![
        ("Header", 0, 64),
        ("Block Table", 64, 512),
    ];
    
    for (i, b) in blocks.iter().enumerate() {
        if b.size > 0 {
            sections.push((BLOCK_NAMES[i], b.offset, b.size));
        }
    }
    
    if tok_size > 0 {
        sections.push(("Tokenizer", tok_offset, tok_size));
    }
    
    sections.push(("Manifest", header.manifest_offset, header.manifest_size));
    
    // Ordenar por offset
    sections.sort_by_key(|s| s.1);
    
    for (name, _offset, size) in &sections {
        let pct = *size as f64 / file_size as f64 * 100.0;
        let bar = make_bar(*size, file_size, 35);
        
        println!("│  {:20} [{}] {:>6.2}%               │", name, bar, pct);
    }
    
    println!("└──────────────────────────────────────────────────────────────────────────────┘");
    println!();
    
    // ═══════════════════════════════════════════════════════════════
    // MANIFEST (opcional)
    // ═══════════════════════════════════════════════════════════════
    if args.manifest && header.manifest_size > 0 {
        f.seek(SeekFrom::Start(header.manifest_offset))?;
        let mut manifest_data = vec![0u8; header.manifest_size as usize];
        f.read_exact(&mut manifest_data)?;
        
        if let Ok(json_str) = String::from_utf8(manifest_data) {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&json_str) {
                println!("┌──────────────────────────────────────────────────────────────────────────────┐");
                println!("│ MANIFEST JSON                                                                │");
                println!("├──────────────────────────────────────────────────────────────────────────────┤");
                let pretty = serde_json::to_string_pretty(&json).unwrap_or_default();
                for line in pretty.lines().take(30) {
                    println!("│  {}  │", format!("{:74}", line));
                }
                if pretty.lines().count() > 30 {
                    println!("│  ... (truncado)                                                              │");
                }
                println!("└──────────────────────────────────────────────────────────────────────────────┘");
            }
        }
    }
    
    Ok(())
}
