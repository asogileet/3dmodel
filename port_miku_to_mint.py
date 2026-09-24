import json
import struct
import numpy as np

def port_miku_to_mint():
    # First re-build clean mint_modular.glb from asset
    from build_mint_modular import build_mint_modular
    build_mint_modular()

    miku_glb = 'z:/home/edwintai/github/3dmodel/miku_modular.glb'
    mint_glb = 'z:/home/edwintai/github/3dmodel/mint_modular.glb'
    dst_glb = 'z:/home/edwintai/github/3dmodel/mint_modular.glb'

    print("Reading miku_modular.glb...")
    with open(miku_glb, 'rb') as f:
        f.read(12)
        chunk_len, chunk_type = struct.unpack('<I4s', f.read(8))
        g_miku = json.loads(f.read(chunk_len).decode('utf-8'))
        bin_len, bin_type = struct.unpack('<I4s', f.read(8))
        b_miku = f.read(bin_len)

    print("Reading mint_modular.glb...")
    with open(mint_glb, 'rb') as f:
        magic, version, length = struct.unpack('<4sII', f.read(12))
        chunk_len, chunk_type = struct.unpack('<I4s', f.read(8))
        g_mint = json.loads(f.read(chunk_len).decode('utf-8'))
        bin_len, bin_type = struct.unpack('<I4s', f.read(8))
        b_mint = bytearray(f.read(bin_len))

    # Helper to extract accessor data from Miku
    def get_acc_data(acc_idx):
        acc = g_miku['accessors'][acc_idx]
        bv = g_miku['bufferViews'][acc['bufferView']]
        off = bv.get('byteOffset', 0) + acc.get('byteOffset', 0)
        cnt = acc['count']
        c_type = acc['componentType']
        t_str = acc['type']

        dtype_map = {
            5126: np.float32,
            5125: np.uint32,
            5123: np.uint16,
            5121: np.uint8
        }
        dim_map = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}
        dt = dtype_map[c_type]
        dim = dim_map[t_str]
        byte_cnt = cnt * dim * dt().itemsize
        arr = np.frombuffer(b_miku[off:off+byte_cnt], dtype=dt)
        if dim > 1:
            arr = arr.reshape(cnt, dim)
        return arr.copy(), c_type, t_str

    # Transformation from Miku scale (neck base Y=17.505, Z=0.967) to Mint neck base (Y=1.379, Z=0.005)
    scale = 0.070
    center_miku = np.array([0.0, 17.505, 0.967], dtype=np.float32)
    center_mint = np.array([0.0, 1.379, 0.005], dtype=np.float32)

    def transform_positions(pos_arr):
        return (pos_arr - center_miku) * scale + center_mint

    def align_mint():
        pad = (4 - (len(b_mint) % 4)) % 4
        if pad > 0:
            b_mint.extend(b'\x00' * pad)

    align_mint()

    # 1. Port Textures 0, 2, 3 from Miku to Mint
    tex_remap = {}
    sampler_idx = 0
    if not g_mint.get('samplers'):
        g_mint['samplers'] = [{'magFilter': 9729, 'minFilter': 9987, 'wrapS': 10497, 'wrapT': 10497}]

    for m_tex in [0, 2, 3]:
        m_img_idx = g_miku['textures'][m_tex]['source']
        m_img = g_miku['images'][m_img_idx]
        m_bv = g_miku['bufferViews'][m_img['bufferView']]
        img_bytes = b_miku[m_bv['byteOffset']:m_bv['byteOffset'] + m_bv['byteLength']]

        align_mint()
        img_off = len(b_mint)
        b_mint.extend(img_bytes)

        mint_bv = len(g_mint['bufferViews'])
        g_mint['bufferViews'].append({
            'buffer': 0,
            'byteOffset': img_off,
            'byteLength': len(img_bytes)
        })

        mint_img = len(g_mint.get('images', []))
        g_mint.setdefault('images', []).append({
            'bufferView': mint_bv,
            'mimeType': m_img.get('mimeType', 'image/png'),
            'name': f'Tex_Miku_{m_tex}'
        })

        mint_tex = len(g_mint.get('textures', []))
        g_mint.setdefault('textures', []).append({
            'sampler': sampler_idx,
            'source': mint_img,
            'name': f'Tex_Miku_{m_tex}'
        })
        tex_remap[m_tex] = mint_tex

    # 2. Port Materials from Miku to Mint
    def add_unlit_mat(name, tex_idx):
        idx = len(g_mint['materials'])
        g_mint['materials'].append({
            'name': name,
            'pbrMetallicRoughness': {
                'baseColorTexture': {'index': tex_idx},
                'metallicFactor': 0.0,
                'roughnessFactor': 0.5
            },
            'extensions': {'KHR_materials_unlit': {}},
            'doubleSided': True
        })
        return idx

    mat_headset = add_unlit_mat('Mat_Miku_Headset', tex_remap[0])
    mat_face = add_unlit_mat('Mat_Miku_Head_Face', tex_remap[2])
    mat_hair = add_unlit_mat('Mat_Miku_Hair_Twintails', tex_remap[3])

    def append_buffer_and_accessor(arr, comp_type, type_str, target=34962):
        align_mint()
        off = len(b_mint)
        data_bytes = arr.tobytes()
        b_mint.extend(data_bytes)

        bv_idx = len(g_mint['bufferViews'])
        g_mint['bufferViews'].append({
            'buffer': 0,
            'byteOffset': off,
            'byteLength': len(data_bytes),
            'target': target
        })

        acc_idx = len(g_mint['accessors'])
        d = {
            'bufferView': bv_idx,
            'componentType': comp_type,
            'count': len(arr),
            'type': type_str
        }
        if type_str in ['SCALAR', 'VEC3']:
            d['min'] = arr.min(axis=0).tolist() if arr.ndim > 1 else [int(arr.min())]
            d['max'] = arr.max(axis=0).tolist() if arr.ndim > 1 else [int(arr.max())]
        g_mint['accessors'].append(d)
        return acc_idx

    # 3. Port Mesh Primitives: Head_Miku_Face
    head_subparts = ['Head_Miku', 'Face_Mouth_Miku', 'Face_Eyes_Left', 'Face_Eyes_Right']
    combined_pos = []
    combined_nor = []
    combined_uvs = []
    combined_idx = []
    combined_jnt = []
    combined_wgt = []
    idx_offset = 0

    for name in head_subparts:
        for node in g_miku['nodes']:
            if node.get('name') == name:
                mesh = g_miku['meshes'][node['mesh']]
                prim = mesh['primitives'][0]
                pos, ct_p, ts_p = get_acc_data(prim['attributes']['POSITION'])
                nor, ct_n, ts_n = get_acc_data(prim['attributes']['NORMAL'])
                uv,  ct_u, ts_u = get_acc_data(prim['attributes']['TEXCOORD_0'])
                idx, ct_i, ts_i = get_acc_data(prim['indices'])

                t_pos = transform_positions(pos)
                cnt = len(pos)

                combined_pos.append(t_pos)
                combined_nor.append(nor)
                combined_uvs.append(uv)
                combined_idx.append(idx.astype(np.uint32) + idx_offset)

                j = np.zeros((cnt, 4), dtype=np.uint16)
                j[:, 0] = 108 # Bip001-Head_0174
                j[:, 1] = 107 # Bip001-Neck_0173
                w = np.zeros((cnt, 4), dtype=np.float32)
                w[:, 0] = 0.95
                w[:, 1] = 0.05
                combined_jnt.append(j)
                combined_wgt.append(w)

                idx_offset += cnt

    face_pos = np.vstack(combined_pos).astype(np.float32)
    face_nor = np.vstack(combined_nor).astype(np.float32)
    face_uvs = np.vstack(combined_uvs).astype(np.float32)
    face_idx = np.concatenate(combined_idx).astype(np.uint32)
    face_jnt = np.vstack(combined_jnt).astype(np.uint16)
    face_wgt = np.vstack(combined_wgt).astype(np.float32)

    acc_face_p = append_buffer_and_accessor(face_pos, 5126, 'VEC3')
    acc_face_n = append_buffer_and_accessor(face_nor, 5126, 'VEC3')
    acc_face_u = append_buffer_and_accessor(face_uvs, 5126, 'VEC2')
    acc_face_i = append_buffer_and_accessor(face_idx, 5125, 'SCALAR', target=34963)
    acc_face_j = append_buffer_and_accessor(face_jnt, 5123, 'VEC4')
    acc_face_w = append_buffer_and_accessor(face_wgt, 5126, 'VEC4')

    mesh_face_idx = len(g_mint['meshes'])
    g_mint['meshes'].append({
        'name': 'Head_Miku_Face',
        'primitives': [{
            'attributes': {
                'POSITION': acc_face_p,
                'NORMAL': acc_face_n,
                'TEXCOORD_0': acc_face_u,
                'JOINTS_0': acc_face_j,
                'WEIGHTS_0': acc_face_w
            },
            'indices': acc_face_i,
            'material': mat_face
        }]
    })

    node_face_idx = len(g_mint['nodes'])
    g_mint['nodes'].append({
        'name': 'Head_Miku_Face',
        'mesh': mesh_face_idx,
        'skin': 3
    })
    g_mint['scenes'][0]['nodes'].append(node_face_idx)
    print(f"Added Head_Miku_Face: {len(face_pos)} vertices, {len(face_idx)//3} triangles.")

    # 4. Port Head_Miku_Cranium (Solid 3D skull cap / back of head)
    for node in g_miku['nodes']:
        if node.get('name') == 'Head_Miku_Cranium':
            mesh = g_miku['meshes'][node['mesh']]
            prim = mesh['primitives'][0]
            pos, _, _ = get_acc_data(prim['attributes']['POSITION'])
            nor, _, _ = get_acc_data(prim['attributes']['NORMAL'])
            uv,  _, _ = get_acc_data(prim['attributes']['TEXCOORD_0'])
            idx, _, _ = get_acc_data(prim['indices'])

            t_pos = transform_positions(pos)
            cnt = len(pos)
            j = np.zeros((cnt, 4), dtype=np.uint16)
            j[:, 0] = 108 # Bip001-Head_0174
            j[:, 1] = 107 # Bip001-Neck_0173
            w = np.zeros((cnt, 4), dtype=np.float32)
            w[:, 0] = 0.95
            w[:, 1] = 0.05

            acc_cr_p = append_buffer_and_accessor(t_pos, 5126, 'VEC3')
            acc_cr_n = append_buffer_and_accessor(nor, 5126, 'VEC3')
            acc_cr_u = append_buffer_and_accessor(uv, 5126, 'VEC2')
            acc_cr_i = append_buffer_and_accessor(idx.astype(np.uint32), 5125, 'SCALAR', target=34963)
            acc_cr_j = append_buffer_and_accessor(j, 5123, 'VEC4')
            acc_cr_w = append_buffer_and_accessor(w, 5126, 'VEC4')

            mesh_cr_idx = len(g_mint['meshes'])
            g_mint['meshes'].append({
                'name': 'Head_Miku_Cranium',
                'primitives': [{
                    'attributes': {
                        'POSITION': acc_cr_p,
                        'NORMAL': acc_cr_n,
                        'TEXCOORD_0': acc_cr_u,
                        'JOINTS_0': acc_cr_j,
                        'WEIGHTS_0': acc_cr_w
                    },
                    'indices': acc_cr_i,
                    'material': mat_hair
                }]
            })

            node_cr_idx = len(g_mint['nodes'])
            g_mint['nodes'].append({
                'name': 'Head_Miku_Cranium',
                'mesh': mesh_cr_idx,
                'skin': 3
            })
            g_mint['scenes'][0]['nodes'].append(node_cr_idx)
            print(f"Added Head_Miku_Cranium: {cnt} vertices, {len(idx)//3} triangles.")

    # 5. Port Hair_Miku_Twintails
    for node in g_miku['nodes']:
        if node.get('name') == 'Hair_Miku_Twintails':
            mesh = g_miku['meshes'][node['mesh']]
            prim = mesh['primitives'][0]
            pos, _, _ = get_acc_data(prim['attributes']['POSITION'])
            nor, _, _ = get_acc_data(prim['attributes']['NORMAL'])
            uv,  _, _ = get_acc_data(prim['attributes']['TEXCOORD_0'])
            idx, _, _ = get_acc_data(prim['indices'])

            t_pos = transform_positions(pos)
            cnt = len(pos)
            j = np.zeros((cnt, 4), dtype=np.uint16)
            j[:, 0] = 108
            w = np.zeros((cnt, 4), dtype=np.float32)
            w[:, 0] = 1.0

            acc_hair_p = append_buffer_and_accessor(t_pos, 5126, 'VEC3')
            acc_hair_n = append_buffer_and_accessor(nor, 5126, 'VEC3')
            acc_hair_u = append_buffer_and_accessor(uv, 5126, 'VEC2')
            acc_hair_i = append_buffer_and_accessor(idx.astype(np.uint32), 5125, 'SCALAR', target=34963)
            acc_hair_j = append_buffer_and_accessor(j, 5123, 'VEC4')
            acc_hair_w = append_buffer_and_accessor(w, 5126, 'VEC4')

            mesh_hair_idx = len(g_mint['meshes'])
            g_mint['meshes'].append({
                'name': 'Hair_Miku_Twintails',
                'primitives': [{
                    'attributes': {
                        'POSITION': acc_hair_p,
                        'NORMAL': acc_hair_n,
                        'TEXCOORD_0': acc_hair_u,
                        'JOINTS_0': acc_hair_j,
                        'WEIGHTS_0': acc_hair_w
                    },
                    'indices': acc_hair_i,
                    'material': mat_hair
                }]
            })

            node_hair_idx = len(g_mint['nodes'])
            g_mint['nodes'].append({
                'name': 'Hair_Miku_Twintails',
                'mesh': mesh_hair_idx,
                'skin': 3
            })
            g_mint['scenes'][0]['nodes'].append(node_hair_idx)
            print(f"Added Hair_Miku_Twintails: {cnt} vertices, {len(idx)//3} triangles.")

    # 7. Port Accessory_Miku_Headset
    for node in g_miku['nodes']:
        if node.get('name') == 'Accessory_Miku_Headset':
            mesh = g_miku['meshes'][node['mesh']]
            prim = mesh['primitives'][0]
            pos, _, _ = get_acc_data(prim['attributes']['POSITION'])
            nor, _, _ = get_acc_data(prim['attributes']['NORMAL'])
            uv,  _, _ = get_acc_data(prim['attributes']['TEXCOORD_0'])
            idx, _, _ = get_acc_data(prim['indices'])

            t_pos = transform_positions(pos)
            cnt = len(pos)
            j = np.zeros((cnt, 4), dtype=np.uint16)
            j[:, 0] = 108
            w = np.zeros((cnt, 4), dtype=np.float32)
            w[:, 0] = 1.0

            acc_hs_p = append_buffer_and_accessor(t_pos, 5126, 'VEC3')
            acc_hs_n = append_buffer_and_accessor(nor, 5126, 'VEC3')
            acc_hs_u = append_buffer_and_accessor(uv, 5126, 'VEC2')
            acc_hs_i = append_buffer_and_accessor(idx.astype(np.uint32), 5125, 'SCALAR', target=34963)
            acc_hs_j = append_buffer_and_accessor(j, 5123, 'VEC4')
            acc_hs_w = append_buffer_and_accessor(w, 5126, 'VEC4')

            mesh_hs_idx = len(g_mint['meshes'])
            g_mint['meshes'].append({
                'name': 'Accessory_Miku_Headset',
                'primitives': [{
                    'attributes': {
                        'POSITION': acc_hs_p,
                        'NORMAL': acc_hs_n,
                        'TEXCOORD_0': acc_hs_u,
                        'JOINTS_0': acc_hs_j,
                        'WEIGHTS_0': acc_hs_w
                    },
                    'indices': acc_hs_i,
                    'material': mat_headset
                }]
            })

            node_hs_idx = len(g_mint['nodes'])
            g_mint['nodes'].append({
                'name': 'Accessory_Miku_Headset',
                'mesh': mesh_hs_idx,
                'skin': 3
            })
            g_mint['scenes'][0]['nodes'].append(node_hs_idx)
            print(f"Added Accessory_Miku_Headset: {cnt} vertices, {len(idx)//3} triangles.")

    # 8. Create Hair_Mint_Back_Bob (the short bob back portion of Mint's hair for short hair mode)
    for node in g_mint['nodes']:
        if node.get('name') == 'Hair_Mint_Back':
            mesh = g_mint['meshes'][node['mesh']]
            prim = mesh['primitives'][0]
            
            # Read Mint's Hair_Mint_Back data
            acc_p = g_mint['accessors'][prim['attributes']['POSITION']]
            bv_p = g_mint['bufferViews'][acc_p['bufferView']]
            off_p = bv_p.get('byteOffset', 0) + acc_p.get('byteOffset', 0)
            m_pos = np.frombuffer(b_mint[off_p:off_p + acc_p['count']*12], dtype=np.float32).reshape(-1, 3)

            acc_i = g_mint['accessors'][prim['indices']]
            bv_i = g_mint['bufferViews'][acc_i['bufferView']]
            off_i = bv_i.get('byteOffset', 0) + acc_i.get('byteOffset', 0)
            c_type_i = acc_i['componentType']
            dt_i = np.uint16 if c_type_i == 5123 else np.uint32
            m_idx = np.frombuffer(b_mint[off_i:off_i + acc_i['count']*dt_i().itemsize], dtype=dt_i).reshape(-1, 3)

            # Extract bob triangles (Y >= 1.15)
            bob_tris = [t for t in m_idx if any(m_pos[v, 1] >= 1.30 for v in t) and all(m_pos[v, 1] >= 1.15 for v in t)]
            bob_tris = np.array(bob_tris)
            bob_v_orig = np.unique(bob_tris)
            remap = {orig: new for new, orig in enumerate(bob_v_orig)}
            bob_idx_new = np.vectorize(remap.get)(bob_tris).astype(np.uint32).flatten()

            # Extract attributes for these vertices
            def get_mint_attr(attr_name, dim, dt):
                acc = g_mint['accessors'][prim['attributes'][attr_name]]
                bv = g_mint['bufferViews'][acc['bufferView']]
                off = bv.get('byteOffset', 0) + acc.get('byteOffset', 0)
                byte_cnt = acc['count'] * dim * dt().itemsize
                arr = np.frombuffer(b_mint[off:off+byte_cnt], dtype=dt)
                if dim > 1:
                    arr = arr.reshape(acc['count'], dim)
                return arr[bob_v_orig]

            bob_pos = get_mint_attr('POSITION', 3, np.float32)
            bob_nor = get_mint_attr('NORMAL', 3, np.float32)
            bob_uv = get_mint_attr('TEXCOORD_0', 2, np.float32)
            bob_jnt = get_mint_attr('JOINTS_0', 4, np.uint16)
            bob_wgt = get_mint_attr('WEIGHTS_0', 4, np.float32)

            acc_mb_p = append_buffer_and_accessor(bob_pos, 5126, 'VEC3')
            acc_mb_n = append_buffer_and_accessor(bob_nor, 5126, 'VEC3')
            acc_mb_u = append_buffer_and_accessor(bob_uv, 5126, 'VEC2')
            acc_mb_i = append_buffer_and_accessor(bob_idx_new, 5125, 'SCALAR', target=34963)
            acc_mb_j = append_buffer_and_accessor(bob_jnt, 5123, 'VEC4')
            acc_mb_w = append_buffer_and_accessor(bob_wgt, 5126, 'VEC4')

            mesh_mb_idx = len(g_mint['meshes'])
            g_mint['meshes'].append({
                'name': 'Hair_Mint_Back_Bob',
                'primitives': [{
                    'attributes': {
                        'POSITION': acc_mb_p,
                        'NORMAL': acc_mb_n,
                        'TEXCOORD_0': acc_mb_u,
                        'JOINTS_0': acc_mb_j,
                        'WEIGHTS_0': acc_mb_w
                    },
                    'indices': acc_mb_i,
                    'material': prim['material']
                }]
            })

            node_mb_idx = len(g_mint['nodes'])
            g_mint['nodes'].append({
                'name': 'Hair_Mint_Back_Bob',
                'mesh': mesh_mb_idx,
                'skin': node.get('skin', 3)
            })
            g_mint['scenes'][0]['nodes'].append(node_mb_idx)
            print(f"Added Hair_Mint_Back_Bob: {len(bob_pos)} vertices, {len(bob_idx_new)//3} triangles.")
            break

    # 9. Repack into mint_modular.glb
    align_mint()
    new_json_bytes = json.dumps(g_mint, separators=(',', ':')).encode('utf-8')
    pad_json = (4 - (len(new_json_bytes) % 4)) % 4
    new_json_bytes += b' ' * pad_json

    total_len = 12 + 8 + len(new_json_bytes) + 8 + len(b_mint)

    with open(dst_glb, 'wb') as f:
        f.write(struct.pack('<4sII', magic, version, total_len))
        f.write(struct.pack('<I4s', len(new_json_bytes), chunk_type))
        f.write(new_json_bytes)
        f.write(struct.pack('<I4s', len(b_mint), bin_type))
        f.write(b_mint)

    size_mb = len(b_mint) / (1024 * 1024)
    print(f"Successfully cross-ported Miku onto Mint body in {dst_glb}! Total buffer size: {size_mb:.2f} MB")

if __name__ == '__main__':
    port_miku_to_mint()
