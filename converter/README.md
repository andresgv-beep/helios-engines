# HELIOS-CONVERT

Conversor de modelos HuggingFace (safetensors) a formato HNFv9.

## Características

- **Cuantización HQS** con grid search MSE paralelo (Rayon)
  - HQ5K: 5-bit, ~4% error
  - HQ4K: 4-bit, ~8% error
  - FP16: Sin pérdida
- **Mixed Precision**: Embeddings y LayerNorms en FP16, pesos en HQxK
- **Multimodal**: Soporte para text, vision, audio
- **HNFv9 compliant**: 16 bloques, execution_hints, manifest

## Instalación

```bash
# Clonar
git clone https://github.com/helios-ai/helios-convert
cd helios-convert

# Build release
cargo build --release

# Instalar (opcional)
cargo install --path .
```

## Uso

```bash
# Básico (HQ5K con MSE, mixed precision)
helios-convert models/qwen2.5-4b -o output/qwen.hnf

# HQ4K (más compresión, más error)
helios-convert models/qwen2.5-4b -o output/qwen.hnf --quant HQ4K

# Sin MSE (rápido, ~27% error)
helios-convert models/qwen2.5-4b -o output/qwen.hnf --no-mse

# FP16 (sin cuantización)
helios-convert models/qwen2.5-4b -o output/qwen.hnf --quant FP16

# Controlar threads
helios-convert models/qwen2.5-4b -o output/qwen.hnf --threads 12

# Verbose
helios-convert models/qwen2.5-4b -o output/qwen.hnf -v
```

## Performance

| Método | Tensor 100M | Error HQ5K | Error HQ4K |
|--------|-------------|------------|------------|
| MSE Grid 64×64 | ~10 seg | ~4% | ~8% |
| Fast (sin MSE) | ~2 seg | ~27% | ~30% |
| Python original | ~30 min | ~4% | ~8% |

## Arquitecturas Soportadas

- LLaMA / LLaMA 2 / LLaMA 3
- Qwen / Qwen2
- Gemma / Gemma 2
- Mistral / Mixtral
- Phi-3
- (Genérico para otros transformers)

## Estructura del Código

```
src/
├── main.rs          # CLI
├── lib.rs           # Exports
├── hnf/             # HNFv9 format
│   ├── header.rs    # 64-byte header
│   └── writer.rs    # File writer
├── hqs/             # Quantization
│   ├── common.rs    # Structures
│   ├── grid_search.rs  # MSE optimization (paralelo)
│   ├── hq4k.rs      # 4-bit
│   └── hq5k.rs      # 5-bit
├── safetensor/      # Reader
├── mapping/         # Canonical names
├── htf/             # Tokenizer
└── hints/           # execution_hints
```

## Licencia

MIT

## Autor

Andrés García / HELIOS Project
