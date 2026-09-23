import json
import struct
import numpy as np

def build_complete_miku():
    miku_src = 'z:/home/edwintai/github/3dmodel/miku_psp_with_bones_fixed.glb'
    avatar_src = 'z:/home/edwintai/github/3dmodel/anime_avatar_modular.glb'
    dst_path = 'z:/home/edwintai/github/3dmodel/miku_modular.glb'

    # 1. Read Miku GLB
    with open(miku_src, 'rb') as f:
        magic, version, length = struct.unpack('<4sII', f.read(12))
        chunk_len, chunk_type = struct.unpack('<I4s', f.read(8))
        miku_gltf = json.loads(f.read(chunk_len).decode('utf-8'))
        bin_len, bin_type = struct.unpack('<I4s', f.read(8))
        miku_bin = bytearray(f.read(bin_len))

    # 2. Read Avatar GLB
    with open(avatar_src, 'rb') as f:
        magic_a, version_a, length_a = struct.unpack('<4sII', f.read(12))
        chunk_len_a, chunk_type_a = struct.unpack('<I4s', f.read(8))
        avatar_gltf = json.loads(f.read(chunk_len_a).decode('utf-8'))
        bin_len_a, bin_type_a = struct.unpack('<I4s', f.read(8))
        avatar_bin = f.read(bin_len_a)

    # 3. Rename Nodes and Meshes in Miku
    rename_nodes = {
        'Object_101': 'Hair_Miku_Twintails',
        'Object_103': 'Head_Miku',
        'Object_100': 'Face_Mouth_Miku',
        'Object_102': 'Accessory_Miku_Tie',
        'Object_96':  'Accessory_Miku_Headset',
        'Object_97':  'Shoes_Miku_Boots',
        'Object_94':  'Outfit_Miku_Sleeves_Left',
        'Object_104': 'Outfit_Miku_Sleeves_Right',
        'Object_98':  'Face_Eyes_Left',
        'Object_99':  'Face_Eyes_Right',
    }
    for node in miku_gltf.get('nodes', []):
        old_name = node.get('name')
        if old_name in rename_nodes:
            node['name'] = rename_nodes[old_name]
    for mesh in miku_gltf.get('meshes', []):
        old_name = mesh.get('name')
        if old_name in rename_nodes:
            mesh['name'] = rename_nodes[old_name]

    # 4. Split Object_95 (Uniform) into Outfit_Miku_Top and Outfit_Miku_Skirt
    # Mesh 1 is Object_95
    mesh1 = miku_gltf['meshes'][1]
    mesh1['name'] = 'Outfit_Miku_Top'
    prim1 = mesh1['primitives'][0]
    
    idx_acc = miku_gltf['accessors'][prim1['indices']]
    bv_i = miku_gltf['bufferViews'][idx_acc['bufferView']]
    off_i = bv_i.get('byteOffset', 0) + idx_acc.get('byteOffset', 0)
    indices = np.frombuffer(miku_bin[off_i:off_i + idx_acc['count']*4], dtype=np.uint32)

    pos_acc = miku_gltf['accessors'][prim1['attributes']['POSITION']]
    bv_p = miku_gltf['bufferViews'][pos_acc['bufferView']]
    off_p = bv_p.get('byteOffset', 0) + pos_acc.get('byteOffset', 0)
    pos = np.frombuffer(miku_bin[off_p:off_p + pos_acc['count']*12], dtype=np.float32).reshape(-1, 3)

    triangles = indices.reshape(-1, 3)
    tri_y = pos[triangles][:, :, 1].mean(axis=1)

    top_tri = triangles[tri_y >= 12.2].flatten()
    skirt_tri = triangles[tri_y < 12.2].flatten()
    print(f"Splitting Uniform: Top has {len(top_tri)} indices, Skirt has {len(skirt_tri)} indices.")

    # Pad current miku_bin to multiple of 4
    if len(miku_bin) % 4 != 0:
        miku_bin.extend(b'\x00' * (4 - (len(miku_bin) % 4)))

    # Append Top indices to miku_bin
    top_bytes = top_tri.astype(np.uint32).tobytes()
    top_bv_offset = len(miku_bin)
    miku_bin.extend(top_bytes)
    if len(miku_bin) % 4 != 0:
        miku_bin.extend(b'\x00' * (4 - (len(miku_bin) % 4)))

    # Append Skirt indices to miku_bin
    skirt_bytes = skirt_tri.astype(np.uint32).tobytes()
    skirt_bv_offset = len(miku_bin)
    miku_bin.extend(skirt_bytes)
    if len(miku_bin) % 4 != 0:
        miku_bin.extend(b'\x00' * (4 - (len(miku_bin) % 4)))

    # Add bufferViews for Top and Skirt indices
    bv_top_idx = len(miku_gltf['bufferViews'])
    miku_gltf['bufferViews'].append({
        'buffer': 0,
        'byteOffset': top_bv_offset,
        'byteLength': len(top_bytes),
        'target': 34963 # ELEMENT_ARRAY_BUFFER
    })

    bv_skirt_idx = len(miku_gltf['bufferViews'])
    miku_gltf['bufferViews'].append({
        'buffer': 0,
        'byteOffset': skirt_bv_offset,
        'byteLength': len(skirt_bytes),
        'target': 34963 # ELEMENT_ARRAY_BUFFER
    })

    # Add accessors for Top and Skirt indices
    acc_top_idx = len(miku_gltf['accessors'])
    miku_gltf['accessors'].append({
        'bufferView': bv_top_idx,
        'componentType': 5125, # UNSIGNED_INT
        'count': len(top_tri),
        'type': 'SCALAR',
        'max': [int(top_tri.max())],
        'min': [int(top_tri.min())]
    })

    acc_skirt_idx = len(miku_gltf['accessors'])
    miku_gltf['accessors'].append({
        'bufferView': bv_skirt_idx,
        'componentType': 5125,
        'count': len(skirt_tri),
        'type': 'SCALAR',
        'max': [int(skirt_tri.max())],
        'min': [int(skirt_tri.min())]
    })

    # Update Mesh 1 (Top) primitive indices
    prim1['indices'] = acc_top_idx

    # Create Mesh Skirt
    skirt_mesh_idx = len(miku_gltf['meshes'])
    skirt_prim = dict(prim1) # copy vertex attributes and material
    skirt_prim['indices'] = acc_skirt_idx
    miku_gltf['meshes'].append({
        'name': 'Outfit_Miku_Skirt',
        'primitives': [skirt_prim]
    })

    # Update Node 95 to Outfit_Miku_Top
    miku_gltf['nodes'][95]['name'] = 'Outfit_Miku_Top'

    # Add Node for Outfit_Miku_Skirt
    skirt_node_idx = len(miku_gltf['nodes'])
    miku_gltf['nodes'].append({
        'name': 'Outfit_Miku_Skirt',
        'mesh': skirt_mesh_idx,
        'skin': 0
    })
    # Add to RootNode (node 2) children
    miku_gltf['nodes'][2]['children'].append(skirt_node_idx)

    # 5. Add Mat_Hair Material for imported anime hairstyles
    mat_hair_idx = len(miku_gltf['materials'])
    miku_gltf['materials'].append({
        'name': 'Mat_Hair',
        'doubleSided': True,
        'pbrMetallicRoughness': {
            'baseColorFactor': [1.0, 1.0, 1.0, 1.0], # White tintable
            'metallicFactor': 0.0,
            'roughnessFactor': 0.4
        }
    })

    # 6. Port 5 Hair meshes from anime_avatar_modular.glb to Miku
    hair_names = ['Hair_Bob', 'Hair_Hime', 'Hair_Parted', 'Hair_Spiky', 'Hair_Twintails']
    # Node 11 is Head in Miku
    head_node = miku_gltf['nodes'][11]
    if 'children' not in head_node:
        head_node['children'] = []

    for h_name in hair_names:
        m_a = [m for m in avatar_gltf['meshes'] if m['name'] == h_name][0]
        p_a = m_a['primitives'][0]

        # Extract positions
        acc_p_a = avatar_gltf['accessors'][p_a['attributes']['POSITION']]
        bv_p_a = avatar_gltf['bufferViews'][acc_p_a['bufferView']]
        off_p_a = bv_p_a.get('byteOffset', 0) + acc_p_a.get('byteOffset', 0)
        pos_h = np.frombuffer(avatar_bin[off_p_a:off_p_a + acc_p_a['count']*12], dtype=np.float32).reshape(-1, 3).copy()

        # Extract normals
        acc_n_a = avatar_gltf['accessors'][p_a['attributes']['NORMAL']]
        bv_n_a = avatar_gltf['bufferViews'][acc_n_a['bufferView']]
        off_n_a = bv_n_a.get('byteOffset', 0) + acc_n_a.get('byteOffset', 0)
        norm_h = np.frombuffer(avatar_bin[off_n_a:off_n_a + acc_n_a['count']*12], dtype=np.float32).reshape(-1, 3).copy()

        # Extract indices
        acc_i_a = avatar_gltf['accessors'][p_a['indices']]
        bv_i_a = avatar_gltf['bufferViews'][acc_i_a['bufferView']]
        off_i_a = bv_i_a.get('byteOffset', 0) + acc_i_a.get('byteOffset', 0)
        c_type = acc_i_a['componentType']
        idx_dtype = np.uint16 if c_type == 5123 else np.uint32
        idx_h = np.frombuffer(avatar_bin[off_i_a:off_i_a + acc_i_a['count']*idx_dtype().itemsize], dtype=idx_dtype).copy()

        # Scale & align to Head local coordinates (Head bone origin is at Y=17.854, Z=0.221)
        # In avatar GLB, Head origin is Y=0.7208, Z=0.0
        # Miku head scale factor ~ 6.2x in width, 7.5x in height, 6.2x in depth
        pos_h[:, 0] = pos_h[:, 0] * 6.2
        pos_h[:, 1] = (pos_h[:, 1] - 0.7208) * 7.5 + 0.35
        pos_h[:, 2] = pos_h[:, 2] * 6.2 + 0.35

        # Append to miku_bin
        if len(miku_bin) % 4 != 0:
            miku_bin.extend(b'\x00' * (4 - (len(miku_bin) % 4)))

        # 1) Positions
        pos_bytes = pos_h.astype(np.float32).tobytes()
        pos_bv_offset = len(miku_bin)
        miku_bin.extend(pos_bytes)
        if len(miku_bin) % 4 != 0: miku_bin.extend(b'\x00' * (4 - (len(miku_bin) % 4)))

        # 2) Normals
        norm_bytes = norm_h.astype(np.float32).tobytes()
        norm_bv_offset = len(miku_bin)
        miku_bin.extend(norm_bytes)
        if len(miku_bin) % 4 != 0: miku_bin.extend(b'\x00' * (4 - (len(miku_bin) % 4)))

        # 3) Indices
        idx_bytes = idx_h.astype(idx_dtype).tobytes()
        idx_bv_offset = len(miku_bin)
        miku_bin.extend(idx_bytes)
        if len(miku_bin) % 4 != 0: miku_bin.extend(b'\x00' * (4 - (len(miku_bin) % 4)))

        # BufferViews
        bv_pos_idx = len(miku_gltf['bufferViews'])
        miku_gltf['bufferViews'].append({
            'buffer': 0,
            'byteOffset': pos_bv_offset,
            'byteLength': len(pos_bytes),
            'target': 34962 # ARRAY_BUFFER
        })

        bv_norm_idx = len(miku_gltf['bufferViews'])
        miku_gltf['bufferViews'].append({
            'buffer': 0,
            'byteOffset': norm_bv_offset,
            'byteLength': len(norm_bytes),
            'target': 34962 # ARRAY_BUFFER
        })

        bv_idx_idx = len(miku_gltf['bufferViews'])
        miku_gltf['bufferViews'].append({
            'buffer': 0,
            'byteOffset': idx_bv_offset,
            'byteLength': len(idx_bytes),
            'target': 34963 # ELEMENT_ARRAY_BUFFER
        })

        # Accessors
        acc_pos_idx = len(miku_gltf['accessors'])
        miku_gltf['accessors'].append({
            'bufferView': bv_pos_idx,
            'componentType': 5126, # FLOAT
            'count': len(pos_h),
            'type': 'VEC3',
            'max': [float(x) for x in pos_h.max(axis=0)],
            'min': [float(x) for x in pos_h.min(axis=0)]
        })

        acc_norm_idx = len(miku_gltf['accessors'])
        miku_gltf['accessors'].append({
            'bufferView': bv_norm_idx,
            'componentType': 5126, # FLOAT
            'count': len(norm_h),
            'type': 'VEC3',
            'max': [float(x) for x in norm_h.max(axis=0)],
            'min': [float(x) for x in norm_h.min(axis=0)]
        })

        acc_indices_idx = len(miku_gltf['accessors'])
        miku_gltf['accessors'].append({
            'bufferView': bv_idx_idx,
            'componentType': c_type,
            'count': len(idx_h),
            'type': 'SCALAR',
            'max': [int(idx_h.max())],
            'min': [int(idx_h.min())]
        })

        # Mesh
        hair_mesh_idx = len(miku_gltf['meshes'])
        miku_gltf['meshes'].append({
            'name': h_name,
            'primitives': [{
                'attributes': {
                    'POSITION': acc_pos_idx,
                    'NORMAL': acc_norm_idx
                },
                'indices': acc_indices_idx,
                'material': mat_hair_idx
            }]
        })

        # Node (as direct child of Head bone, Node 11)
        hair_node_idx = len(miku_gltf['nodes'])
        miku_gltf['nodes'].append({
            'name': h_name,
            'mesh': hair_mesh_idx
        })
        head_node['children'].append(hair_node_idx)
        print(f"Added {h_name} to Head node (mesh {hair_mesh_idx}, node {hair_node_idx}).")

    # Update buffer 0 byteLength
    miku_gltf['buffers'][0]['byteLength'] = len(miku_bin)

    # 7. Write out new binary GLB
    new_json_bytes = json.dumps(miku_gltf, separators=(',', ':')).encode('utf-8')
    json_padding = (4 - (len(new_json_bytes) % 4)) % 4
    new_json_bytes += b' ' * json_padding
    new_chunk_len = len(new_json_bytes)

    bin_padding = (4 - (len(miku_bin) % 4)) % 4
    miku_bin.extend(b'\x00' * bin_padding)
    new_bin_len = len(miku_bin)

    total_len = 12 + 8 + new_chunk_len + 8 + new_bin_len

    with open(dst_path, 'wb') as f:
        f.write(struct.pack('<4sII', magic, version, total_len))
        f.write(struct.pack('<I4s', new_chunk_len, chunk_type))
        f.write(new_json_bytes)
        f.write(struct.pack('<I4s', new_bin_len, bin_type))
        f.write(miku_bin)

    print(f"Successfully created complete modular Miku {dst_path} ({total_len/1024:.1f} KB)!")

if __name__ == '__main__':
    build_complete_miku()
