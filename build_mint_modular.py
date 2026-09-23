import json
import struct
import os

def build_mint_modular():
    src_path = 'z:/home/edwintai/github/3dmodel/assets/mint_swimsuit_animated-_neverness_to_everness.glb'
    dst_path = 'z:/home/edwintai/github/3dmodel/mint_modular.glb'

    print(f"Reading {src_path}...")
    with open(src_path, 'rb') as f:
        magic, version, length = struct.unpack('<4sII', f.read(12))
        chunk_len, chunk_type = struct.unpack('<I4s', f.read(8))
        gltf = json.loads(f.read(chunk_len).decode('utf-8'))
        bin_len, bin_type = struct.unpack('<I4s', f.read(8))
        bin_data = f.read(bin_len)

    # Rename map for Nodes and Meshes
    node_renames = {
        'Object_9':   'Accessory_Mint_ChestBow',
        'Object_64':  'Accessory_Mint_Hat',
        'Object_115': 'Accessory_Mint_Ribbon',
        'Object_146': 'Hair_Mint_Back',
        'Object_147': 'Face_Mint_Mask',
        'Object_148': 'Head_Mint_Face',
        'Object_149': 'Face_Mint_Eyelashes',
        'Object_150': 'Face_Mint_Eyes',
        'Object_151': 'Face_Mint_Highlight',
        'Object_152': 'Hair_Mint_Front',
        'Object_153': 'Outfit_Mint_Swimsuit',
        'Object_154': 'Outfit_Mint_Skirt',
        'Object_155': 'Outfit_Mint_Accessories',
    }

    mesh_renames = {
        0:  'Accessory_Mint_ChestBow',
        1:  'Accessory_Mint_Hat',
        2:  'Accessory_Mint_Ribbon',
        3:  'Hair_Mint_Back',
        4:  'Face_Mint_Mask',
        5:  'Head_Mint_Face',
        6:  'Face_Mint_Eyelashes',
        7:  'Face_Mint_Eyes',
        8:  'Face_Mint_Highlight',
        9:  'Hair_Mint_Front',
        10: 'Outfit_Mint_Swimsuit',
        11: 'Outfit_Mint_Skirt',
        12: 'Outfit_Mint_Accessories',
    }

    for node in gltf.get('nodes', []):
        old_n = node.get('name')
        if old_n in node_renames:
            node['name'] = node_renames[old_n]

    for idx, mesh in enumerate(gltf.get('meshes', [])):
        if idx in mesh_renames:
            mesh['name'] = mesh_renames[idx]

    # Ensure Animation name is clean and recognized
    for anim in gltf.get('animations', []):
        anim['name'] = 'idle' # Standard name for animation system

    # Ensure KHR_materials_unlit is active on all materials
    if 'KHR_materials_unlit' not in gltf.get('extensionsUsed', []):
        gltf.setdefault('extensionsUsed', []).append('KHR_materials_unlit')

    for mat in gltf.get('materials', []):
        mat.setdefault('extensions', {})['KHR_materials_unlit'] = {}

    # Pack into new GLB
    new_json_bytes = json.dumps(gltf, separators=(',', ':')).encode('utf-8')
    json_padding = (4 - (len(new_json_bytes) % 4)) % 4
    new_json_bytes += b' ' * json_padding
    new_chunk_len = len(new_json_bytes)

    bin_padding = (4 - (len(bin_data) % 4)) % 4
    new_bin_data = bin_data + (b'\x00' * bin_padding)
    new_bin_len = len(new_bin_data)

    total_len = 12 + 8 + new_chunk_len + 8 + new_bin_len

    with open(dst_path, 'wb') as f:
        f.write(struct.pack('<4sII', magic, version, total_len))
        f.write(struct.pack('<I4s', new_chunk_len, chunk_type))
        f.write(new_json_bytes)
        f.write(struct.pack('<I4s', new_bin_len, bin_type))
        f.write(new_bin_data)

    size_mb = os.path.getsize(dst_path) / (1024 * 1024)
    print(f"Successfully generated {dst_path} ({size_mb:.2f} MB) with clean modular names and KHR_materials_unlit!")

if __name__ == '__main__':
    build_mint_modular()
