#!/usr/bin/env python3
"""
HELIOS HNF Validator v9.0.5
Valida estructura de archivos HNFv9

v9.0.5: Soporte prefijo text. para consistencia de modalidades
"""

import struct
import json
import sys
from pathlib import Path

# ============================================================================
# CONSTANTS
# ============================================================================

MAGIC = b'HNFv9\x00\x00\x00'
HEADER_SIZE = 64
BLOCK_TABLE_ENTRY_SIZE = 32
BLOCK_COUNT = 16

BLOCK_NAMES = [
    "text_model",    # 0x0
    "vision",        # 0x1
    "audio",         # 0x2
    "video",         # 0x3
    "spatial_3d",    # 0x4
    "personality",   # 0x5
    "memory",        # 0x6
    "cortex",        # 0x7
    "code_exec",     # 0x8
    "tools",         # 0x9
    "exec_hints",    # 0xA
    "expert_router", # 0xB
    "reserved_0",    # 0xC
    "reserved_1",    # 0xD
    "reserved_2",    # 0xE
    "reserved_3",    # 0xF
]

# ============================================================================
# READER
# ============================================================================

def read_header(f):
    """Lee header HNFv9 (64 bytes)"""
    data = f.read(64)
    
    magic = data[0:8]
    version_major = struct.unpack('<H', data[8:10])[0]
    version_minor = struct.unpack('<H', data[10:12])[0]
    flags = struct.unpack('<I', data[12:16])[0]
    block_count = struct.unpack('<I', data[16:20])[0]
    header_size = struct.unpack('<I', data[20:24])[0]
    block_table_offset = struct.unpack('<Q', data[24:32])[0]
    manifest_offset = struct.unpack('<Q', data[32:40])[0]
    manifest_size = struct.unpack('<Q', data[40:48])[0]
    file_size = struct.unpack('<Q', data[48:56])[0]
    checksum = struct.unpack('<I', data[56:60])[0]
    
    return {
        'magic': magic,
        'version': f"{version_major}.{version_minor}",
        'flags': flags,
        'block_count': block_count,
        'header_size': header_size,
        'block_table_offset': block_table_offset,
        'manifest_offset': manifest_offset,
        'manifest_size': manifest_size,
        'file_size': file_size,
        'checksum': checksum,
    }

def read_block_table(f):
    """Lee block table (16 x 32 bytes)"""
    blocks = []
    for i in range(16):
        data = f.read(32)
        block_id = struct.unpack('<I', data[0:4])[0]
        block_type = struct.unpack('<I', data[4:8])[0]
        offset = struct.unpack('<Q', data[8:16])[0]
        size = struct.unpack('<Q', data[16:24])[0]
        checksum = struct.unpack('<Q', data[24:32])[0]
        
        blocks.append({
            'id': block_id,
            'type': block_type,
            'name': BLOCK_NAMES[i],
            'offset': offset,
            'size': size,
            'checksum': checksum,
        })
    return blocks

def read_manifest(f, offset, size):
    """Lee manifest JSON"""
    f.seek(offset)
    data = f.read(size)
    return json.loads(data.decode('utf-8'))

def read_execution_hints(f, blocks):
    """Lee execution_hints (bloque 0xA)"""
    block = blocks[0xA]
    if block['size'] == 0:
        return None
    
    f.seek(block['offset'])
    data = f.read(block['size'])
    return json.loads(data.decode('utf-8'))

# ============================================================================
# VALIDATION
# ============================================================================

def validate_hnf(path: str):
    """Valida archivo HNF completo"""
    
    print("═" * 70)
    print(f"  HELIOS HNF VALIDATOR v9.0.5")
    print("═" * 70)
    print(f"  File: {path}")
    print(f"  Size: {Path(path).stat().st_size / 1024 / 1024:.1f} MB")
    print("═" * 70)
    
    errors = []
    warnings = []
    
    with open(path, 'rb') as f:
        # ════════════════════════════════════════════════════════════════
        # HEADER
        # ════════════════════════════════════════════════════════════════
        print("\n[HEADER]")
        header = read_header(f)
        
        # Validar magic
        if header['magic'] != MAGIC:
            errors.append(f"Invalid magic: {header['magic']}")
            print(f"  ✗ Magic: {header['magic']} (expected {MAGIC})")
        else:
            print(f"  ✓ Magic: {header['magic']}")
        
        print(f"  ✓ Version: {header['version']}")
        print(f"  ✓ Flags: 0x{header['flags']:08X}")
        print(f"  ✓ Blocks: {header['block_count']}")
        print(f"  ✓ Manifest: offset={header['manifest_offset']}, size={header['manifest_size']}")
        
        # ════════════════════════════════════════════════════════════════
        # BLOCK TABLE
        # ════════════════════════════════════════════════════════════════
        print("\n[BLOCK TABLE]")
        blocks = read_block_table(f)
        
        active_blocks = []
        for b in blocks:
            if b['size'] > 0:
                active_blocks.append(b)
                size_mb = b['size'] / 1024 / 1024
                print(f"  ✓ [{b['id']:X}] {b['name']:15} : {size_mb:8.2f} MB @ offset {b['offset']}")
        
        if not active_blocks:
            errors.append("No active blocks found")
        
        # ════════════════════════════════════════════════════════════════
        # EXECUTION HINTS
        # ════════════════════════════════════════════════════════════════
        print("\n[EXECUTION HINTS]")
        hints = read_execution_hints(f, blocks)
        
        if hints is None:
            warnings.append("No execution_hints block")
            print("  ⚠ No execution_hints found")
        else:
            # v9.0.5: Verificar TEXT (ahora en sección separada)
            if hints.get('text_enabled'):
                print(f"\n  [TEXT CONFIG]")
                tc = hints.get('text', {})
                print(f"    ✓ arch: {tc.get('arch', 'N/A')}")
                print(f"    ✓ num_hidden_layers: {tc.get('num_hidden_layers', 'N/A')}")
                print(f"    ✓ hidden_size: {tc.get('hidden_size', 'N/A')}")
                print(f"    ✓ vocab_size: {tc.get('vocab_size', 'N/A')}")
                print(f"    ✓ attention_type: {tc.get('attention_type', 'N/A')}")
                print(f"    ✓ qkv_layout: {tc.get('qkv_layout', 'N/A')}")
                print(f"    ✓ mlp_type: {tc.get('mlp_type', 'N/A')}")
                
                if 'rope_scaling' in tc:
                    rs = tc['rope_scaling']
                    print(f"    ✓ rope_scaling: type={rs.get('type', 'N/A')}")
            else:
                # Fallback: hints en raíz (formato antiguo)
                print(f"  ✓ arch: {hints.get('arch', 'N/A')}")
                print(f"  ✓ num_hidden_layers: {hints.get('num_hidden_layers', 'N/A')}")
                print(f"  ✓ hidden_size: {hints.get('hidden_size', 'N/A')}")
                print(f"  ✓ vocab_size: {hints.get('vocab_size', 'N/A')}")
                print(f"  ✓ attention_type: {hints.get('attention_type', 'N/A')}")
                print(f"  ✓ qkv_layout: {hints.get('qkv_layout', 'N/A')}")
                print(f"  ✓ mlp_type: {hints.get('mlp_type', 'N/A')}")
                
                if 'rope_scaling' in hints:
                    rs = hints['rope_scaling']
                    print(f"  ✓ rope_scaling: type={rs.get('type', 'N/A')}")
            
            # Verificar VISION
            if hints.get('vision_enabled'):
                print(f"\n  [VISION CONFIG]")
                vc = hints.get('vision_config', {})
                print(f"    ✓ arch: {vc.get('arch', 'N/A')}")
                print(f"    ✓ image_size: {vc.get('image_size', 'N/A')}")
                print(f"    ✓ patch_size: {vc.get('patch_size', 'N/A')}")
            
            # Verificar CORTEX
            if hints.get('cortex_enabled'):
                print(f"\n  [CORTEX CONFIG]")
                cc = hints.get('cortex', {})
                print(f"    ✓ arch: {cc.get('arch', 'N/A')}")
                print(f"    ✓ num_hidden_layers: {cc.get('num_hidden_layers', 'N/A')}")
                print(f"    ✓ hidden_size: {cc.get('hidden_size', 'N/A')}")
                print(f"    ✓ vocab_size: {cc.get('vocab_size', 'N/A')}")
                print(f"    ✓ qkv_layout: {cc.get('qkv_layout', 'N/A')}")
                print(f"    ✓ mlp_type: {cc.get('mlp_type', 'N/A')}")
                print(f"    ✓ partial_rotary_factor: {cc.get('partial_rotary_factor', 'N/A')}")
                
                if 'rope_scaling' in cc:
                    rs = cc['rope_scaling']
                    print(f"    ✓ rope_scaling: type={rs.get('type', 'N/A')}")
                    if 'long_factor' in rs:
                        print(f"    ✓ long_factor: [{len(rs['long_factor'])} values]")
            
            # Verificar CODE
            if hints.get('code_enabled'):
                print(f"\n  [CODE CONFIG]")
                cc = hints.get('code', {})
                print(f"    ✓ arch: {cc.get('arch', 'N/A')}")
                print(f"    ✓ num_hidden_layers: {cc.get('num_hidden_layers', 'N/A')}")
                print(f"    ✓ hidden_size: {cc.get('hidden_size', 'N/A')}")
                print(f"    ✓ vocab_size: {cc.get('vocab_size', 'N/A')}")
                
                if 'rope_scaling' in cc:
                    rs = cc['rope_scaling']
                    print(f"    ✓ rope_scaling: type={rs.get('type', 'N/A')}, factor={rs.get('factor', 'N/A')}")
        
        # ════════════════════════════════════════════════════════════════
        # MANIFEST (TENSORS)
        # ════════════════════════════════════════════════════════════════
        print("\n[MANIFEST]")
        manifest = read_manifest(f, header['manifest_offset'], header['manifest_size'])
        
        tensors = manifest.get('tensors', [])
        print(f"  ✓ Total tensors: {len(tensors)}")
        
        # Contar por bloque
        by_block = {}
        for t in tensors:
            block = t.get('block', 'unknown')
            by_block[block] = by_block.get(block, 0) + 1
        
        for block, count in sorted(by_block.items()):
            print(f"    - {block}: {count} tensors")
        
        # ════════════════════════════════════════════════════════════════
        # PREFIX CHECK (v9.0.5: Todas las modalidades tienen prefijo)
        # ════════════════════════════════════════════════════════════════
        print("\n  [PREFIX CHECK]")
        prefixes = {
            'text_model': [],
            'vision': [],
            'cortex': [],
            'code_exec': [],
            'audio': [],
        }
        
        for t in tensors:
            name = t.get('name', '')
            block = t.get('block', '')
            
            if block in prefixes:
                prefixes[block].append(name)
        
        # v9.0.5: Verificar que TEXT tiene prefijo text.
        text_tensors = prefixes.get('text_model', [])
        if text_tensors:
            has_prefix = all(n.startswith('text.') for n in text_tensors)
            sample = text_tensors[0] if text_tensors else 'N/A'
            if has_prefix:
                print(f"  ✓ TEXT: prefijo 'text.' OK (sample: {sample})")
            else:
                # Warning, no error - puede ser formato antiguo
                warnings.append("TEXT tensors missing 'text.' prefix (old format?)")
                print(f"  ⚠ TEXT: falta prefijo 'text.' (sample: {sample})")
        
        # Verificar que CORTEX tiene prefijo cortex.
        cortex_tensors = prefixes.get('cortex', [])
        if cortex_tensors:
            has_prefix = all(n.startswith('cortex.') for n in cortex_tensors)
            sample = cortex_tensors[0] if cortex_tensors else 'N/A'
            if has_prefix:
                print(f"  ✓ CORTEX: prefijo 'cortex.' OK (sample: {sample})")
            else:
                warnings.append("CORTEX tensors missing 'cortex.' prefix")
                print(f"  ⚠ CORTEX: falta prefijo 'cortex.' (sample: {sample})")
        
        # Verificar que CODE tiene prefijo code.
        code_tensors = prefixes.get('code_exec', [])
        if code_tensors:
            has_prefix = all(n.startswith('code.') for n in code_tensors)
            sample = code_tensors[0] if code_tensors else 'N/A'
            if has_prefix:
                print(f"  ✓ CODE: prefijo 'code.' OK (sample: {sample})")
            else:
                warnings.append("CODE tensors missing 'code.' prefix")
                print(f"  ⚠ CODE: falta prefijo 'code.' (sample: {sample})")
        
        # Verificar que VISION tiene prefijo vision.
        vision_tensors = prefixes.get('vision', [])
        if vision_tensors:
            has_prefix = all(n.startswith('vision.') for n in vision_tensors)
            sample = vision_tensors[0] if vision_tensors else 'N/A'
            if has_prefix:
                print(f"  ✓ VISION: prefijo 'vision.' OK (sample: {sample})")
            else:
                warnings.append("VISION tensors missing 'vision.' prefix")
                print(f"  ⚠ VISION: falta prefijo 'vision.' (sample: {sample})")
        
        # Verificar que AUDIO tiene prefijo audio.
        audio_tensors = prefixes.get('audio', [])
        if audio_tensors:
            has_prefix = all(n.startswith('audio.') for n in audio_tensors)
            sample = audio_tensors[0] if audio_tensors else 'N/A'
            if has_prefix:
                print(f"  ✓ AUDIO: prefijo 'audio.' OK (sample: {sample})")
            else:
                warnings.append("AUDIO tensors missing 'audio.' prefix")
                print(f"  ⚠ AUDIO: falta prefijo 'audio.' (sample: {sample})")
        
        # ════════════════════════════════════════════════════════════════
        # SUMMARY
        # ════════════════════════════════════════════════════════════════
        print("\n" + "═" * 70)
        if errors:
            print("  ✗ VALIDATION FAILED")
            for e in errors:
                print(f"    ERROR: {e}")
        elif warnings:
            print("  ⚠ VALIDATION PASSED WITH WARNINGS")
            for w in warnings:
                print(f"    WARNING: {w}")
        else:
            print("  ✓ VALIDATION PASSED")
        print("═" * 70)
        
        return len(errors) == 0

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_hnf.py <file.hnf>")
        sys.exit(1)
    
    path = sys.argv[1]
    if not Path(path).exists():
        print(f"Error: File not found: {path}")
        sys.exit(1)
    
    success = validate_hnf(path)
    sys.exit(0 if success else 1)
