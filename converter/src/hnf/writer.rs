// src/hnf/writer.rs
// ============================================================================
// HNF WRITER - Construye archivos HNFv9
// ============================================================================

use std::fs::File;
use std::io::{BufWriter, Write, Seek, SeekFrom};
use std::path::Path;

use anyhow::{Context, Result};
use xxhash_rust::xxh3::{xxh3_64, Xxh3};

use super::header::*;

/// Información de un tensor para el manifest
#[derive(Debug, Clone, serde::Serialize)]
pub struct TensorManifest {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub offset: u64,
    pub size: u64,
    pub numel: usize,
}

/// Builder para archivos HNFv9
pub struct HnfWriter {
    file: BufWriter<File>,
    header: HnfHeader,
    block_table: BlockTable,
    current_offset: u64,
    tensor_manifests: Vec<Vec<TensorManifest>>,  // Por bloque
    block_hashers: Vec<Option<Xxh3>>,  // Hasher incremental por bloque
}

impl HnfWriter {
    /// Crea un nuevo archivo HNF
    pub fn create(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::create(path.as_ref())
            .with_context(|| format!("Cannot create {}", path.as_ref().display()))?;
        let mut file = BufWriter::new(file);
        
        let header = HnfHeader::default();
        let block_table = BlockTable::default();
        
        // Escribir header placeholder
        file.write_all(&header.to_bytes())?;
        
        // Escribir block table placeholder
        file.write_all(&block_table.to_bytes())?;
        
        // Offset actual: después de header + block table
        let current_offset = HEADER_SIZE as u64 + 512;
        
        // Inicializar manifests vacíos para cada bloque
        let tensor_manifests = (0..16).map(|_| Vec::new()).collect();
        
        // Inicializar hashers como None
        let block_hashers = (0..16).map(|_| None).collect();
        
        Ok(Self {
            file,
            header,
            block_table,
            current_offset,
            tensor_manifests,
            block_hashers,
        })
    }
    
    /// Alinea el offset actual a múltiplo de 32
    fn align_32(&mut self) -> Result<()> {
        let remainder = self.current_offset % 32;
        if remainder != 0 {
            let padding = 32 - remainder;
            let zeros = vec![0u8; padding as usize];
            self.file.write_all(&zeros)?;
            self.current_offset += padding;
        }
        Ok(())
    }
    
    /// Escribe datos de un bloque
    pub fn write_block(&mut self, block_id: usize, data: &[u8]) -> Result<()> {
        if block_id >= 16 {
            anyhow::bail!("Invalid block_id: {}", block_id);
        }
        
        // Alinear
        self.align_32()?;
        
        // Guardar offset inicial
        let block_offset = self.current_offset;
        
        // Escribir datos
        self.file.write_all(data)?;
        
        // Calcular checksum XXH3-64
        let checksum = xxh3_64(data);
        
        // Actualizar block table
        self.block_table.entries[block_id].offset = block_offset;
        self.block_table.entries[block_id].size = data.len() as u64;
        self.block_table.entries[block_id].checksum = checksum;
        
        // Actualizar offset
        self.current_offset += data.len() as u64;
        
        Ok(())
    }
    
    /// Escribe un tensor cuantizado a un bloque específico
    pub fn write_tensor(
        &mut self,
        block_id: usize,
        name: &str,
        dtype: &str,
        shape: &[usize],
        data: &[u8],
    ) -> Result<()> {
        if block_id >= 16 {
            anyhow::bail!("Invalid block_id: {}", block_id);
        }
        
        // Si es el primer tensor del bloque, marcar offset e inicializar hasher
        if self.block_table.entries[block_id].size == 0 {
            self.align_32()?;
            self.block_table.entries[block_id].offset = self.current_offset;
            self.block_hashers[block_id] = Some(Xxh3::new());
        }
        
        let tensor_offset = self.current_offset;
        
        // Escribir datos
        self.file.write_all(data)?;
        self.current_offset += data.len() as u64;
        
        // Actualizar hasher incremental
        if let Some(ref mut hasher) = self.block_hashers[block_id] {
            hasher.update(data);
        }
        
        // Actualizar size del bloque
        self.block_table.entries[block_id].size = 
            self.current_offset - self.block_table.entries[block_id].offset;
        
        // Añadir al manifest
        let numel: usize = shape.iter().product();
        self.tensor_manifests[block_id].push(TensorManifest {
            name: name.to_string(),
            dtype: dtype.to_string(),
            shape: shape.to_vec(),
            offset: tensor_offset,
            size: data.len() as u64,
            numel,
        });
        
        Ok(())
    }
    
    /// Finaliza un bloque (calcula checksum con el hasher incremental)
    pub fn finalize_block(&mut self, block_id: usize) -> Result<()> {
        if block_id >= 16 {
            return Ok(());
        }
        
        let entry = &self.block_table.entries[block_id];
        if entry.size == 0 {
            return Ok(());  // Bloque vacío
        }
        
        // Calcular checksum desde el hasher incremental
        if let Some(hasher) = self.block_hashers[block_id].take() {
            let checksum = hasher.digest();
            self.block_table.entries[block_id].checksum = checksum;
        }
        
        Ok(())
    }
    
    /// Escribe execution_hints (bloque 0xA)
    pub fn write_execution_hints(&mut self, hints: &serde_json::Value) -> Result<()> {
        let json = serde_json::to_vec(hints)?;
        self.write_block(BLOCK_EXEC_HINTS, &json)?;
        Ok(())
    }
    
    /// Escribe tokenizer HTF (bloque 0x9 - BLOCK_TOKENIZER)
    pub fn write_tokenizer(&mut self, htf_data: &[u8]) -> Result<()> {
        self.write_block(BLOCK_TOKENIZER, htf_data)?;
        Ok(())
    }
    
    /// Finaliza el archivo escribiendo manifest y actualizando header
    pub fn finalize(mut self, mut manifest: serde_json::Value) -> Result<()> {
        // Alinear antes del manifest
        self.align_32()?;
        
        // Construir lista de tensores para el manifest
        let tensor_list: Vec<serde_json::Value> = self.tensor_manifests
            .iter()
            .enumerate()
            .flat_map(|(block_id, tensors)| {
                let block_name = BLOCK_NAMES[block_id];
                tensors.iter().map(move |t| serde_json::json!({
                    "name": t.name,
                    "block": block_name,
                    "offset": t.offset,
                    "size": t.size,
                    "dtype": t.dtype,
                    "shape": t.shape,
                }))
            })
            .collect();
        
        // Añadir tensores al manifest
        if let Some(obj) = manifest.as_object_mut() {
            obj.insert("tensors".to_string(), serde_json::Value::Array(tensor_list));
        }
        
        // Escribir manifest
        let manifest_offset = self.current_offset;
        let manifest_bytes = serde_json::to_vec_pretty(&manifest)?;
        self.file.write_all(&manifest_bytes)?;
        let manifest_size = manifest_bytes.len() as u64;
        
        // Calcular tamaño total
        let file_size = self.current_offset + manifest_size;
        
        // Calcular CRC32 (simplificado - sobre header + block table)
        let checksum = {
            let mut data = self.header.to_bytes();
            data.extend(self.block_table.to_bytes());
            crc32fast::hash(&data)
        };
        
        // Actualizar header
        self.header.manifest_offset = manifest_offset;
        self.header.manifest_size = manifest_size;
        self.header.file_size = file_size;
        self.header.checksum = checksum;
        
        // Actualizar flags basado en bloques no vacíos
        if self.block_table.entries[BLOCK_VISION].size > 0 {
            self.header.flags.set(HeaderFlags::HAS_VISION);
            self.header.flags.set(HeaderFlags::IS_MULTIMODAL);
        }
        if self.block_table.entries[BLOCK_AUDIO].size > 0 {
            self.header.flags.set(HeaderFlags::HAS_AUDIO);
            self.header.flags.set(HeaderFlags::IS_MULTIMODAL);
        }
        if self.block_table.entries[BLOCK_VIDEO].size > 0 {
            self.header.flags.set(HeaderFlags::HAS_VIDEO);
            self.header.flags.set(HeaderFlags::IS_MULTIMODAL);
        }
        if self.block_table.entries[BLOCK_SPATIAL_3D].size > 0 {
            self.header.flags.set(HeaderFlags::HAS_SPATIAL);
            self.header.flags.set(HeaderFlags::IS_MULTIMODAL);
        }
        if self.block_table.entries[BLOCK_PERSONALITY].size > 0 {
            self.header.flags.set(HeaderFlags::HAS_PERSONALITY);
        }
        if self.block_table.entries[BLOCK_MEMORY].size > 0 {
            self.header.flags.set(HeaderFlags::HAS_MEMORY);
        }
        if self.block_table.entries[BLOCK_CORTEX].size > 0 {
            self.header.flags.set(HeaderFlags::HAS_CORTEX);
        }
        if self.block_table.entries[BLOCK_CODE_EXEC].size > 0 {
            self.header.flags.set(HeaderFlags::HAS_CODE_EXEC);
        }
        if self.block_table.entries[BLOCK_TOOLS].size > 0 {
            self.header.flags.set(HeaderFlags::HAS_TOOLS);
        }
        if self.block_table.entries[BLOCK_EXPERT_ROUTER].size > 0 {
            self.header.flags.set(HeaderFlags::HAS_EXPERT_ROUTER);
        }
        
        // Reescribir header al inicio
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(&self.header.to_bytes())?;
        
        // Reescribir block table
        self.file.write_all(&self.block_table.to_bytes())?;
        
        // Flush
        self.file.flush()?;
        
        Ok(())
    }
    
    /// Obtiene manifests de tensores por bloque
    pub fn tensor_manifests(&self) -> &Vec<Vec<TensorManifest>> {
        &self.tensor_manifests
    }
}
