import json
import struct
import numpy as np

def build_miku_cranium():
    src_glb = 'z:/home/edwintai/github/3dmodel/miku_modular.glb'
    dst_glb = 'z:/home/edwintai/github/3dmodel/miku_modular.glb'
    avatar_glb = 'z:/home/edwintai/github/3dmodel/anime_avatar_modular.glb'

    print("Reading miku_modular.glb...")
    with open(src_glb, 'rb') as f:
        magic, version, length = struct.unpack('<4sII', f.read(12))
        chunk_len, chunk_type = struct.unpack('<I4s', f.read(8))
        g = json.loads(f.read(chunk_len).decode('utf-8'))
        bin_len, bin_type = struct.unpack('<I4s', f.read(8))
        b = bytearray(f.read(bin_len))

    print("Reading anime_avatar_modular.glb for Hair_Bob source...")
    with open(avatar_glb, 'rb') as f:
        f.read(12)
        c_len, _ = struct.unpack('<I4s', f.read(8))
        g_avatar = json.loads(f.read(c_len).decode('utf-8'))
        b_len, _ = struct.unpack('<I4s', f.read(8))
        b_avatar = f.read(b_len)

    def align_b():
        pad = (4 - (len(b) % 4)) % 4
        if pad > 0:
            b.extend(b'\x00' * pad)

    def append_data(arr, comp_type, type_str, target=34962):
        align_b()
        off = len(b)
        data_bytes = arr.tobytes()
        b.extend(data_bytes)

        bv_idx = len(g['bufferViews'])
        g['bufferViews'].append({
            'buffer': 0,
            'byteOffset': off,
            'byteLength': len(data_bytes),
            'target': target
        })

        acc_idx = len(g['accessors'])
        d = {
            'bufferView': bv_idx,
            'componentType': comp_type,
            'count': len(arr),
            'type': type_str
        }
        if type_str in ['SCALAR', 'VEC3']:
            d['min'] = arr.min(axis=0).tolist() if arr.ndim > 1 else [int(arr.min())]
            d['max'] = arr.max(axis=0).tolist() if arr.ndim > 1 else [int(arr.max())]
        g['accessors'].append(d)
        return acc_idx

    # 1. Extract Head_Miku position accessor to locate the rim vertices
    head_mesh_idx = None
    for m_i, m in enumerate(g['meshes']):
        if m.get('name') == '9' or m.get('name') == 'Head_Miku':
            head_mesh_idx = m_i
            break

    prim_head = g['meshes'][head_mesh_idx]['primitives'][0]
    acc_p = g['accessors'][prim_head['attributes']['POSITION']]
    bv_p = g['bufferViews'][acc_p['bufferView']]
    off_p = bv_p.get('byteOffset', 0) + acc_p.get('byteOffset', 0)
    pos_head = np.frombuffer(b[off_p:], dtype=np.float32, count=acc_p['count']*3).reshape(-1, 3)

    # Rim vertex indices in counter-clockwise order around cranial opening
    # from nape (X=0) through right temple, forehead apex (X=0), left temple, back to nape
    rim_indices = [346, 308, 322, 329, 331, 332, 344, 361, 362, 363, 364, 366, 355, 352, 351, 321, 318, 311, 349]
    rim_positions = pos_head[rim_indices].copy()

    # 2. Procedurally construct smooth 3D cranium (skull cap, occiput, crown, and temples)
    c = np.array([0.0, 19.12, -0.05], dtype=np.float32)
    pole = np.array([0.0, 19.10, -1.26], dtype=np.float32)

    num_rings = 6
    all_rings = [rim_positions]

    for r in range(1, num_rings):
        t = r / num_rings
        ring = []
        for i, p in enumerate(rim_positions):
            base = (1.0 - t) * p + t * pole
            d = base - c
            dist = np.linalg.norm(d)
            if dist > 1e-4:
                d = d / dist
            target_r = 1.18 + 0.12 * np.sin(t * np.pi)
            cur_p = c + d * target_r
            blend = (1.0 - t * 0.7) * cur_p + (t * 0.7) * base
            ring.append(blend)
        all_rings.append(np.array(ring, dtype=np.float32))

    all_rings.append(np.array([pole], dtype=np.float32))

    cranium_pos = np.vstack(all_rings).astype(np.float32)

    cranium_tris = []
    for r in range(num_rings - 1):
        r1 = r * 19
        r2 = (r + 1) * 19
        for i in range(18):
            cranium_tris.append([r1 + i, r2 + i, r2 + i + 1])
            cranium_tris.append([r1 + i, r2 + i + 1, r1 + i + 1])

    pole_idx = num_rings * 19
    r_last = (num_rings - 1) * 19
    for i in range(18):
        cranium_tris.append([r_last + i, pole_idx, r_last + i + 1])

    cranium_idx = np.array(cranium_tris, dtype=np.uint16).flatten()

    # 3. Compute outward normals
    cranium_nor = np.zeros_like(cranium_pos)
    for t in cranium_tris:
        v0 = cranium_pos[t[0]]
        v1 = cranium_pos[t[1]]
        v2 = cranium_pos[t[2]]
        n = np.cross(v1 - v0, v2 - v0)
        cranium_nor[t[0]] += n
        cranium_nor[t[1]] += n
        cranium_nor[t[2]] += n

    lens = np.linalg.norm(cranium_nor, axis=1, keepdims=True)
    lens[lens == 0] = 1.0
    cranium_nor = (cranium_nor / lens).astype(np.float32)

    # 4. Generate UVs for scalp
    # Spherical / planar UV projection mapped into hair/scalp texture area
    cranium_uvs = np.zeros((len(cranium_pos), 2), dtype=np.float32)
    for i, p in enumerate(cranium_pos):
        u = 0.5 + p[0] / 3.0
        v = 1.0 + (p[1] - 17.5) / 3.0
        cranium_uvs[i] = [np.clip(u, 0.05, 0.95), np.clip(v, 1.05, 1.95)]

    # 5. Skin cranium to Miku's Head bone (joint 11)
    # Node 11 is Head
    # In Miku's skin, find joint index for Node 11
    miku_skin = g['skins'][0]
    head_joint_idx = miku_skin['joints'].index(11)
    print(f"Miku skin joint index for Head (Node 11): {head_joint_idx}")

    cnt_cr = len(cranium_pos)
    cr_jnt = np.zeros((cnt_cr, 4), dtype=np.uint16)
    cr_jnt[:, 0] = head_joint_idx
    cr_wgt = np.zeros((cnt_cr, 4), dtype=np.float32)
    cr_wgt[:, 0] = 1.0

    # Append cranium accessors
    acc_cr_p = append_data(cranium_pos, 5126, 'VEC3')
    acc_cr_n = append_data(cranium_nor, 5126, 'VEC3')
    acc_cr_u = append_data(cranium_uvs, 5126, 'VEC2')
    acc_cr_i = append_data(cranium_idx, 5123, 'SCALAR', target=34963)
    acc_cr_j = append_data(cr_jnt, 5123, 'VEC4')
    acc_cr_w = append_data(cr_wgt, 5126, 'VEC4')

    # Material: Mat 7 is Miku hair material
    mesh_cr_idx = len(g['meshes'])
    g['meshes'].append({
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
            'material': 7
        }]
    })

    node_cr_idx = len(g['nodes'])
    g['nodes'].append({
        'name': 'Head_Miku_Cranium',
        'mesh': mesh_cr_idx,
        'skin': 0
    })
    # Add to Object_5 (parent of all skinned meshes)
    g['nodes'][5]['children'].append(node_cr_idx)
    print(f"Added Head_Miku_Cranium (node {node_cr_idx}, mesh {mesh_cr_idx}) with {cnt_cr} verts, {len(cranium_idx)//3} tris.")

    # 6. Extract Hair_Bob from anime_avatar_modular.glb, scale and skin properly to Miku's head!
    m_bob = [m for m in g_avatar['meshes'] if m['name'] == 'Hair_Bob'][0]
    prim_b = m_bob['primitives'][0]

    acc_p_b = g_avatar['accessors'][prim_b['attributes']['POSITION']]
    bv_p_b = g_avatar['bufferViews'][acc_p_b['bufferView']]
    off_p_b = bv_p_b.get('byteOffset', 0) + acc_p_b.get('byteOffset', 0)
    pos_bob = np.frombuffer(b_avatar[off_p_b:off_p_b + acc_p_b['count']*12], dtype=np.float32).reshape(-1, 3).copy()

    acc_n_b = g_avatar['accessors'][prim_b['attributes']['NORMAL']]
    bv_n_b = g_avatar['bufferViews'][acc_n_b['bufferView']]
    off_n_b = bv_n_b.get('byteOffset', 0) + acc_n_b.get('byteOffset', 0)
    nor_bob = np.frombuffer(b_avatar[off_n_b:off_n_b + acc_n_b['count']*12], dtype=np.float32).reshape(-1, 3).copy()

    acc_i_b = g_avatar['accessors'][prim_b['indices']]
    bv_i_b = g_avatar['bufferViews'][acc_i_b['bufferView']]
    off_i_b = bv_i_b.get('byteOffset', 0) + acc_i_b.get('byteOffset', 0)
    c_type_b = acc_i_b['componentType']
    idx_dt_b = np.uint16 if c_type_b == 5123 else np.uint32
    idx_bob = np.frombuffer(b_avatar[off_i_b:off_i_b + acc_i_b['count']*idx_dt_b().itemsize], dtype=idx_dt_b).copy()

    # Scale and center onto Miku's head
    scale = 6.2
    pos_bob[:, 0] = pos_bob[:, 0] * scale
    pos_bob[:, 1] = (pos_bob[:, 1] - 0.8845) * scale + 19.12
    pos_bob[:, 2] = (pos_bob[:, 2] - 0.007) * scale + 0.05

    # UVs for Hair_Bob
    uv_bob = np.zeros((len(pos_bob), 2), dtype=np.float32)
    for i, p in enumerate(pos_bob):
        u = 0.5 + p[0] / 3.0
        v = 1.0 + (p[1] - 17.5) / 3.0
        uv_bob[i] = [np.clip(u, 0.05, 0.95), np.clip(v, 1.05, 1.95)]

    # Skin Hair_Bob to Head bone (joint 11)
    cnt_bob = len(pos_bob)
    bob_jnt = np.zeros((cnt_bob, 4), dtype=np.uint16)
    bob_jnt[:, 0] = head_joint_idx
    bob_wgt = np.zeros((cnt_bob, 4), dtype=np.float32)
    bob_wgt[:, 0] = 1.0

    acc_bob_p = append_data(pos_bob.astype(np.float32), 5126, 'VEC3')
    acc_bob_n = append_data(nor_bob.astype(np.float32), 5126, 'VEC3')
    acc_bob_u = append_data(uv_bob.astype(np.float32), 5126, 'VEC2')
    acc_bob_i = append_data(idx_bob.astype(np.uint16), 5123, 'SCALAR', target=34963)
    acc_bob_j = append_data(bob_jnt, 5123, 'VEC4')
    acc_bob_w = append_data(bob_wgt, 5126, 'VEC4')

    mesh_bob_idx = len(g['meshes'])
    g['meshes'].append({
        'name': 'Hair_Miku_Short',
        'primitives': [{
            'attributes': {
                'POSITION': acc_bob_p,
                'NORMAL': acc_bob_n,
                'TEXCOORD_0': acc_bob_u,
                'JOINTS_0': acc_bob_j,
                'WEIGHTS_0': acc_bob_w
            },
            'indices': acc_bob_i,
            'material': 7
        }]
    })

    node_bob_idx = len(g['nodes'])
    g['nodes'].append({
        'name': 'Hair_Miku_Short',
        'mesh': mesh_bob_idx,
        'skin': 0
    })
    g['nodes'][5]['children'].append(node_bob_idx)
    print(f"Added Hair_Miku_Short (node {node_bob_idx}, mesh {mesh_bob_idx}) with {cnt_bob} verts, {len(idx_bob)//3} tris.")

    # Update buffers byteLength
    align_b()
    g['buffers'][0]['byteLength'] = len(b)

    # Repack GLB
    new_json_bytes = json.dumps(g, separators=(',', ':')).encode('utf-8')
    pad_json = (4 - (len(new_json_bytes) % 4)) % 4
    new_json_bytes += b' ' * pad_json

    total_len = 12 + 8 + len(new_json_bytes) + 8 + len(b)

    with open(dst_glb, 'wb') as f:
        f.write(struct.pack('<4sII', magic, version, total_len))
        f.write(struct.pack('<I4s', len(new_json_bytes), chunk_type))
        f.write(new_json_bytes)
        f.write(struct.pack('<I4s', len(b), bin_type))
        f.write(b)

    size_kb = total_len / 1024
    print(f"Successfully wrote {dst_glb} ({size_kb:.1f} KB) with Head_Miku_Cranium and Hair_Miku_Short!")

if __name__ == '__main__':
    build_miku_cranium()
