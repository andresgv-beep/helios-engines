// src/hnf/header.rs
// ============================================================================
// HNFv9 HEADER - Estructura de 64 bytes
// ============================================================================

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Read, Write, Cursor};

/// Magic bytes para HNFv9
pub const MAGIC: &[u8; 8] = b"HNFv9\x00\x00\x00";

/// Versión actual - HNFv9.1
pub const VERSION_MAJOR: u16 = 9;
pub const VERSION_MINOR: u16 = 1;

/// Número fijo de bloques
pub const BLOCK_COUNT: u32 = 16;

/// Tamaño del header
pub const HEADER_SIZE: u32 = 64;

/// Índices de bloques - HNFv9.1
pub const BLOCK_TEXT_MODEL: usize = 0x0;
pub const BLOCK_VISION: usize = 0x1;
pub const BLOCK_AUDIO: usize = 0x2;
pub const BLOCK_VIDEO: usize = 0x3;
pub const BLOCK_SPATIAL_3D: usize = 0x4;
pub const BLOCK_PERSONALITY: usize = 0x5;
pub const BLOCK_MEMORY: usize = 0x6;
pub const BLOCK_CORTEX: usize = 0x7;
pub const BLOCK_CODE_EXEC: usize = 0x8;
pub const BLOCK_TOKENIZER: usize = 0x9;       // HTF multi-domain
pub const BLOCK_EXEC_HINTS: usize = 0xA;      // JSON (obligatorio)
pub const BLOCK_EXEC_HINTS_BIN: usize = 0xB;  // Binario (preferido) ← NUEVO
pub const BLOCK_TOOLS: usize = 0xC;
pub const BLOCK_EXPERT_ROUTER: usize = 0xD;
pub const BLOCK_RESERVED_0: usize = 0xE;
pub const BLOCK_RESERVED_1: usize = 0xF;

/// Nombres de bloques
pub const BLOCK_NAMES: [&str; 16] = [
    "text_model",        // 0x0
    "vision",            // 0x1
    "audio",             // 0x2
    "video",             // 0x3
    "spatial_3d",        // 0x4
    "personality",       // 0x5
    "memory",            // 0x6
    "cortex",            // 0x7
    "code_exec",         // 0x8
    "tokenizer",         // 0x9
    "execution_hints",   // 0xA
    "exec_hints_bin",    // 0xB ← NUEVO
    "tools",             // 0xC
    "expert_router",     // 0xD
    "reserved_0",        // 0xE
    "reserved_1",        // 0xF
];

/// Flags del header
#[derive(Debug, Clone, Copy, Default)]
pub struct HeaderFlags(pub u32);

impl HeaderFlags {
    pub const HAS_VISION: u32 = 1 << 0;
    pub const HAS_AUDIO: u32 = 1 << 1;
    pub const HAS_VIDEO: u32 = 1 << 2;
    pub const HAS_SPATIAL: u32 = 1 << 3;
    pub const HAS_PERSONALITY: u32 = 1 << 4;
    pub const HAS_MEMORY: u32 = 1 << 5;
    pub const HAS_CORTEX: u32 = 1 << 6;
    pub const HAS_CODE_EXEC: u32 = 1 << 7;
    pub const HAS_TOKENIZER: u32 = 1 << 8;       // ← NUEVO
    pub const HAS_EXEC_HINTS_BIN: u32 = 1 << 9;  // ← NUEVO
    pub const HAS_TOOLS: u32 = 1 << 10;
    pub const HAS_EXPERT_ROUTER: u32 = 1 << 11;
    pub const IS_MOE: u32 = 1 << 12;
    pub const IS_MULTIMODAL: u32 = 1 << 13;
    
    pub fn set(&mut self, flag: u32) {
        self.0 |= flag;
    }
    
    pub fn has(&self, flag: u32) -> bool {
        (self.0 & flag) != 0
    }
}

/// Header HNFv9 (64 bytes)
#[derive(Debug, Clone)]
pub struct HnfHeader {
    pub magic: [u8; 8],
    pub version_major: u16,
    pub version_minor: u16,
    pub flags: HeaderFlags,
    pub block_count: u32,
    pub header_size: u32,
    pub block_table_offset: u64,
    pub manifest_offset: u64,
    pub manifest_size: u64,
    pub file_size: u64,
    pub checksum: u32,
    pub reserved: u32,
}

impl Default for HnfHeader {
    fn default() -> Self {
        Self {
            magic: *MAGIC,
            version_major: VERSION_MAJOR,
            version_minor: VERSION_MINOR,
            flags: HeaderFlags::default(),
            block_count: BLOCK_COUNT,
            header_size: HEADER_SIZE,
            block_table_offset: HEADER_SIZE as u64,
            manifest_offset: 0,
            manifest_size: 0,
            file_size: 0,
            checksum: 0,
            reserved: 0,
        }
    }
}

impl HnfHeader {
    /// Serializa a bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(64);
        buf.extend_from_slice(&self.magic);
        buf.write_u16::<LittleEndian>(self.version_major).unwrap();
        buf.write_u16::<LittleEndian>(self.version_minor).unwrap();
        buf.write_u32::<LittleEndian>(self.flags.0).unwrap();
        buf.write_u32::<LittleEndian>(self.block_count).unwrap();
        buf.write_u32::<LittleEndian>(self.header_size).unwrap();
        buf.write_u64::<LittleEndian>(self.block_table_offset).unwrap();
        buf.write_u64::<LittleEndian>(self.manifest_offset).unwrap();
        buf.write_u64::<LittleEndian>(self.manifest_size).unwrap();
        buf.write_u64::<LittleEndian>(self.file_size).unwrap();
        buf.write_u32::<LittleEndian>(self.checksum).unwrap();
        buf.write_u32::<LittleEndian>(self.reserved).unwrap();
        buf
    }
    
    /// Deserializa desde bytes
    pub fn from_bytes(data: &[u8]) -> std::io::Result<Self> {
        let mut cursor = Cursor::new(data);
        
        let mut magic = [0u8; 8];
        cursor.read_exact(&mut magic)?;
        
        Ok(Self {
            magic,
            version_major: cursor.read_u16::<LittleEndian>()?,
            version_minor: cursor.read_u16::<LittleEndian>()?,
            flags: HeaderFlags(cursor.read_u32::<LittleEndian>()?),
            block_count: cursor.read_u32::<LittleEndian>()?,
            header_size: cursor.read_u32::<LittleEndian>()?,
            block_table_offset: cursor.read_u64::<LittleEndian>()?,
            manifest_offset: cursor.read_u64::<LittleEndian>()?,
            manifest_size: cursor.read_u64::<LittleEndian>()?,
            file_size: cursor.read_u64::<LittleEndian>()?,
            checksum: cursor.read_u32::<LittleEndian>()?,
            reserved: cursor.read_u32::<LittleEndian>()?,
        })
    }
    
    /// Valida el header
    pub fn validate(&self) -> Result<(), String> {
        if &self.magic != MAGIC {
            return Err(format!("Invalid magic: {:?}", self.magic));
        }
        if self.version_major != VERSION_MAJOR {
            return Err(format!("Unsupported version: {}.{}", self.version_major, self.version_minor));
        }
        if self.block_count != BLOCK_COUNT {
            return Err(format!("Invalid block count: {} (expected {})", self.block_count, BLOCK_COUNT));
        }
        if self.header_size != HEADER_SIZE {
            return Err(format!("Invalid header size: {}", self.header_size));
        }
        Ok(())
    }
}

/// Entrada de la Block Table (32 bytes)
#[derive(Debug, Clone, Default)]
pub struct BlockEntry {
    pub block_id: u32,
    pub block_type: u32,
    pub offset: u64,
    pub size: u64,
    pub checksum: u64,
}

impl BlockEntry {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(32);
        buf.write_u32::<LittleEndian>(self.block_id).unwrap();
        buf.write_u32::<LittleEndian>(self.block_type).unwrap();
        buf.write_u64::<LittleEndian>(self.offset).unwrap();
        buf.write_u64::<LittleEndian>(self.size).unwrap();
        buf.write_u64::<LittleEndian>(self.checksum).unwrap();
        buf
    }
    
    pub fn from_bytes(data: &[u8]) -> std::io::Result<Self> {
        let mut cursor = Cursor::new(data);
        Ok(Self {
            block_id: cursor.read_u32::<LittleEndian>()?,
            block_type: cursor.read_u32::<LittleEndian>()?,
            offset: cursor.read_u64::<LittleEndian>()?,
            size: cursor.read_u64::<LittleEndian>()?,
            checksum: cursor.read_u64::<LittleEndian>()?,
        })
    }
    
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
}

/// Block Table completa (16 × 32 = 512 bytes)
pub struct BlockTable {
    pub entries: [BlockEntry; 16],
}

impl Default for BlockTable {
    fn default() -> Self {
        Self {
            entries: std::array::from_fn(|i| BlockEntry {
                block_id: i as u32,
                block_type: i as u32,
                ..Default::default()
            }),
        }
    }
}

impl BlockTable {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(512);
        for entry in &self.entries {
            buf.extend(entry.to_bytes());
        }
        buf
    }
    
    pub fn from_bytes(data: &[u8]) -> std::io::Result<Self> {
        let mut entries: [BlockEntry; 16] = Default::default();
        for i in 0..16 {
            let start = i * 32;
            entries[i] = BlockEntry::from_bytes(&data[start..start + 32])?;
        }
        Ok(Self { entries })
    }
}
