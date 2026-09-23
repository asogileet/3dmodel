import json
import struct

def rename_miku_glb():
    src_path = 'z:/home/edwintai/github/3dmodel/miku_psp_with_bones_fixed.glb'
    dst_path = 'z:/home/edwintai/github/3dmodel/miku_modular.glb'

    with open(src_path, 'rb') as f:
        magic, version, length = struct.unpack('<4sII', f.read(12))
        chunk_len, chunk_type = struct.unpack('<I4s', f.read(8))
        json_bytes = f.read(chunk_len)
        gltf = json.loads(json_bytes.decode('utf-8'))
        
        bin_len, bin_type = struct.unpack('<I4s', f.read(8))
        bin_data = f.read(bin_len)

    # Rename map for Nodes
    rename_nodes = {
        'Object_101': 'Hair_Miku_Twintails',
        'Object_103': 'Head_Miku',
        'Object_100': 'Face_Mouth_Miku',
        'Object_102': 'Accessory_Miku_Tie',
        'Object_96':  'Accessory_Miku_Headset',
        'Object_97':  'Shoes_Miku_Boots',
        'Object_95':  'Outfit_Miku_Uniform',
        'Object_94':  'Outfit_Miku_Sleeves_Left',
        'Object_104': 'Outfit_Miku_Sleeves_Right',
        'Object_98':  'Face_Eyes_Left',
        'Object_99':  'Face_Eyes_Right',
    }

    renamed_count = 0
    for node in gltf.get('nodes', []):
        old_name = node.get('name')
        if old_name in rename_nodes:
            node['name'] = rename_nodes[old_name]
            renamed_count += 1
            print(f"Renamed node {old_name} -> {node['name']}")

    for mesh in gltf.get('meshes', []):
        old_name = mesh.get('name')
        if old_name in rename_nodes:
            mesh['name'] = rename_nodes[old_name]

    print(f"Renamed {renamed_count} nodes in GLTF structure.")

    # Re-pack GLB
    new_json_bytes = json.dumps(gltf, separators=(',', ':')).encode('utf-8')
    # Pad JSON with spaces to multiple of 4
    json_padding = (4 - (len(new_json_bytes) % 4)) % 4
    new_json_bytes += b' ' * json_padding
    new_chunk_len = len(new_json_bytes)

    # Pad BIN if needed
    bin_padding = (4 - (len(bin_data) % 4)) % 4
    new_bin_data = bin_data + (b'\x00' * bin_padding)
    new_bin_len = len(new_bin_data)

    total_len = 12 + 8 + new_chunk_len + 8 + new_bin_len

    with open(dst_path, 'wb') as f:
        # Header
        f.write(struct.pack('<4sII', magic, version, total_len))
        # Chunk 0: JSON
        f.write(struct.pack('<I4s', new_chunk_len, chunk_type))
        f.write(new_json_bytes)
        # Chunk 1: BIN
        f.write(struct.pack('<I4s', new_bin_len, bin_type))
        f.write(new_bin_data)

    print(f"Successfully created clean {dst_path} ({total_len / 1024:.1f} KB)!")

if __name__ == '__main__':
    rename_miku_glb()
