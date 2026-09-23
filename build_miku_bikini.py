import json
import struct
import numpy as np
import zlib

def make_png(width, height, get_pixel):
    raw = bytearray()
    for y in range(height):
        raw.append(0) # Filter 0
        for x in range(width):
            r, g, b, a = get_pixel(x, y, width, height)
            raw.extend([r, g, b, a])
    compressed = zlib.compress(bytes(raw), 9)
    def chunk(tag, data):
        c = tag + data
        return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)
    ihdr = struct.pack('>IIBBBBB', width, height, 8, 6, 0, 0, 0)
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', ihdr) + chunk(b'IDAT', compressed) + chunk(b'IEND', b'')

def generate_bikini_texture():
    W, H = 256, 256
    # Texture layout:
    # U=0.5 is front center, U=0 or U=1 is back center
    # V=0 is bottom (Y=9.59), V=1 is top (Y=17.44)
    def get_pixel(x, y, w, h):
        # image y goes 0 at top to H-1 at bottom
        # V goes 0 at bottom to 1 at top
        u = x / (w - 1)
        v = (h - 1 - y) / (h - 1)

        # Base porcelain anime skin
        r, g, b = 255, 240, 234

        # Subtle clavicle shadow around v=0.88, front
        dist_u_front = abs(u - 0.5)
        if 0.86 <= v <= 0.89 and dist_u_front < 0.25:
            r, g, b = 246, 222, 212

        # Navel (belly button) at v=0.38, u=0.5
        if 0.37 <= v <= 0.39 and dist_u_front < 0.02:
            r, g, b = 225, 195, 185

        # Bikini Top (v between 0.52 and 0.72)
        # Cups are centered at u=0.42 and u=0.58
        is_bikini_top = False
        is_trim = False
        # Front cups
        for cup_u in [0.41, 0.59]:
            du = abs(u - cup_u)
            dv = abs(v - 0.61)
            # Oval cup
            dist = np.sqrt((du / 0.11)**2 + (dv / 0.09)**2)
            if dist <= 1.0:
                is_bikini_top = True
                if dist >= 0.88:
                    is_trim = True
        # Neck halter strap (connecting top of cups to neck)
        if 0.70 <= v <= 0.85:
            if abs(u - 0.44) < 0.015 or abs(u - 0.56) < 0.015:
                is_trim = True
        # Back strap around v=0.61
        if 0.59 <= v <= 0.63:
            is_trim = True

        # Bikini Bottom (v between 0.12 and 0.28)
        is_bikini_bottom = False
        # Front triangle: wider at top, narrower at bottom
        if 0.14 <= v <= 0.27 and dist_u_front < (0.05 + (v - 0.14) * 0.9):
            is_bikini_bottom = True
            if v >= 0.25 or dist_u_front > (0.04 + (v - 0.14) * 0.85):
                is_trim = True
        # Back triangle: u near 0 or 1
        dist_u_back = min(u, 1.0 - u)
        if 0.14 <= v <= 0.27 and dist_u_back < (0.06 + (v - 0.14) * 1.1):
            is_bikini_bottom = True
            if v >= 0.25 or dist_u_back > (0.05 + (v - 0.14) * 1.0):
                is_trim = True
        # Side strings
        if 0.24 <= v <= 0.27:
            is_trim = True

        if is_trim:
            # Dark charcoal / black border with neon green accent
            return (24, 24, 28, 255)
        elif is_bikini_top or is_bikini_bottom:
            # Hatsune Miku iconic vibrant teal: #33c7df -> (51, 199, 223)
            return (51, 199, 223, 255)

        return (r, g, b, 255)

    return make_png(W, H, get_pixel)

def build_miku_bikini():
    src_glb = 'z:/home/edwintai/github/3dmodel/miku_modular.glb'
    dst_glb = 'z:/home/edwintai/github/3dmodel/miku_modular.glb'

    print("Reading miku_modular.glb...")
    with open(src_glb, 'rb') as f:
        magic, version, length = struct.unpack('<4sII', f.read(12))
        chunk_len, chunk_type = struct.unpack('<I4s', f.read(8))
        gltf = json.loads(f.read(chunk_len).decode('utf-8'))
        bin_len, bin_type = struct.unpack('<I4s', f.read(8))
        miku_bin = bytearray(f.read(bin_len))

    # 1. Generate Bikini Texture PNG
    bikini_png = generate_bikini_texture()
    print(f"Generated Bikini PNG: {len(bikini_png)} bytes")

    # Align miku_bin to 4 bytes
    if len(miku_bin) % 4 != 0:
        miku_bin.extend(b'\x00' * (4 - (len(miku_bin) % 4)))

    # Append PNG to binary buffer
    png_offset = len(miku_bin)
    miku_bin.extend(bikini_png)
    if len(miku_bin) % 4 != 0:
        miku_bin.extend(b'\x00' * (4 - (len(miku_bin) % 4)))

    bv_png = len(gltf['bufferViews'])
    gltf['bufferViews'].append({
        'buffer': 0,
        'byteOffset': png_offset,
        'byteLength': len(bikini_png)
    })

    img_idx = len(gltf.get('images', []))
    gltf.setdefault('images', []).append({
        'bufferView': bv_png,
        'mimeType': 'image/png',
        'name': 'Tex_Miku_Bikini'
    })

    sampler_idx = 0
    if not gltf.get('samplers'):
        gltf['samplers'] = [{'magFilter': 9729, 'minFilter': 9987, 'wrapS': 10497, 'wrapT': 10497}]

    tex_idx = len(gltf.get('textures', []))
    gltf.setdefault('textures', []).append({
        'sampler': sampler_idx,
        'source': img_idx,
        'name': 'Tex_Miku_Bikini'
    })

    # Create Material
    mat_idx = len(gltf['materials'])
    gltf['materials'].append({
        'name': 'Mat_Miku_Bikini',
        'pbrMetallicRoughness': {
            'baseColorTexture': {'index': tex_idx},
            'metallicFactor': 0.05,
            'roughnessFactor': 0.55
        },
        'doubleSided': True
    })

    # 2. Build Torso & Limb Geometry
    # Joints in skin 0:
    # 4: Neck, 3: Chest, 2: Spine, 66: Hips, 18: LeftShoulder, 19: LeftUpperArm, 42: RightShoulder, 43: RightUpperArm, 77: LeftUpperLeg, 82: RightUpperLeg
    positions = []
    normals = []
    uvs = []
    joints = []
    weights = []
    indices = []

    # Torso rings
    N_Y = 28
    N_THETA = 24
    y_min, y_max = 9.59, 17.44
    y_coords = np.linspace(y_min, y_max, N_Y)

    for i, y in enumerate(y_coords):
        v = (y - y_min) / (y_max - y_min)

        # Profile parameters
        if y < 10.8: # Thighs / lower pelvis
            t = (y - 9.59) / (10.8 - 9.59)
            rx = 1.70 - t * 0.15
            rz = 1.25 - t * 0.20
            cz = 0.20
        elif y < 12.8: # Pelvis to waist
            t = (y - 10.8) / (12.8 - 10.8)
            rx = 1.55 - t * 0.45 # waist narrows to 1.10
            rz = 1.05 - t * 0.30 # waist narrows to 0.75
            cz = 0.20 + t * 0.10
        elif y < 15.2: # Waist to bust
            t = (y - 12.8) / (15.2 - 12.8)
            rx = 1.10 + t * 0.35 # bust width 1.45
            rz = 0.75 + t * 0.35 # bust depth 1.10
            cz = 0.30 + t * 0.20
        elif y < 16.5: # Bust to shoulders/collar
            t = (y - 15.2) / (16.5 - 15.2)
            rx = 1.45 - t * 0.35
            rz = 1.10 - t * 0.25
            cz = 0.50 + t * 0.30
        else: # Neck
            t = (y - 16.5) / (17.44 - 16.5)
            rx = 1.10 - t * 0.35 # neck rx = 0.75
            rz = 0.85 - t * 0.01 # neck rz = 0.84
            cz = 0.80 + t * 0.27 # neck cz = 1.07 (seamless match to Head_Miku!)

        # Bone weighting for this ring
        if y >= 16.8: # Neck
            j_list = [4, 3, 0, 0] # Neck, Chest
            w_list = [0.85, 0.15, 0.0, 0.0]
        elif y >= 14.5: # Chest
            t_blend = (y - 14.5) / (16.8 - 14.5)
            j_list = [3, 4, 2, 0] # Chest, Neck, Spine
            w_list = [1.0 - t_blend * 0.3, t_blend * 0.3, 0.0, 0.0]
        elif y >= 12.5: # Spine
            t_blend = (y - 12.5) / (14.5 - 12.5)
            j_list = [2, 3, 66, 0] # Spine, Chest, Hips
            w_list = [1.0 - t_blend * 0.4, t_blend * 0.4, 0.0, 0.0]
        elif y >= 10.8: # Hips
            t_blend = (y - 10.8) / (12.5 - 10.8)
            j_list = [66, 2, 0, 0] # Hips, Spine
            w_list = [1.0 - t_blend * 0.4, t_blend * 0.4, 0.0, 0.0]
        else: # Legs/Hips transition
            j_list = [66, 77, 82, 0]
            w_list = [0.6, 0.2, 0.2, 0.0]

        for j in range(N_THETA):
            theta = 2.0 * np.pi * j / N_THETA
            u = j / float(N_THETA)

            cos_t = np.cos(theta)
            sin_t = np.sin(theta)

            # Front bust bump around y in [14.0, 15.6] and sin_t > 0
            front_boost = 0.0
            if 14.0 <= y <= 15.6 and sin_t > 0.2:
                u_dist = abs(cos_t)
                if 0.15 <= u_dist <= 0.75:
                    y_factor = np.sin(np.pi * (y - 14.0) / 1.6)
                    x_factor = np.sin(np.pi * (u_dist - 0.15) / 0.60)
                    front_boost = y_factor * x_factor * 0.48

            px = rx * cos_t
            py = y
            pz = cz + (rz + front_boost) * sin_t

            ring_joints = list(j_list)
            ring_weights = list(w_list)
            if y < 10.8:
                if px > 0.2:
                    ring_joints = [66, 77, 2, 0]
                    ring_weights = [0.4, 0.6, 0.0, 0.0]
                elif px < -0.2:
                    ring_joints = [66, 82, 2, 0]
                    ring_weights = [0.4, 0.6, 0.0, 0.0]

            positions.append([px, py, pz])
            nx = cos_t
            ny = 0.05
            nz = sin_t
            n_len = np.sqrt(nx*nx + ny*ny + nz*nz)
            normals.append([nx/n_len, ny/n_len, nz/n_len])
            uvs.append([u, v])
            joints.append(ring_joints)
            weights.append(ring_weights)

    # Generate ring triangles
    for i in range(N_Y - 1):
        for j in range(N_THETA):
            next_j = (j + 1) % N_THETA
            p1 = i * N_THETA + j
            p2 = (i + 1) * N_THETA + j
            p3 = (i + 1) * N_THETA + next_j
            p4 = i * N_THETA + next_j

            indices.extend([p1, p2, p3])
            indices.extend([p1, p3, p4])

    # 3. Add Left and Right Bare Arms
    # Left Arm
    arm_steps = 10
    arm_radial = 12
    l_start = np.array([1.4, 16.0, 0.3])
    l_end   = np.array([6.8, 12.8, 0.25])
    r_arm = 0.40

    base_idx_l = len(positions)
    for s in range(arm_steps):
        t = s / (arm_steps - 1)
        center = l_start + t * (l_end - l_start)
        v = 0.5 + t * 0.4
        for r in range(arm_radial):
            theta = 2.0 * np.pi * r / arm_radial
            u = r / float(arm_radial)
            p = center + np.array([0, r_arm * np.sin(theta), r_arm * np.cos(theta)])
            positions.append(list(p))
            normals.append([0, np.sin(theta), np.cos(theta)])
            uvs.append([u, v])
            joints.append([18, 19, 3, 0])
            weights.append([1.0 - t * 0.8, t * 0.8, 0.0, 0.0])

    for s in range(arm_steps - 1):
        for r in range(arm_radial):
            next_r = (r + 1) % arm_radial
            p1 = base_idx_l + s * arm_radial + r
            p2 = base_idx_l + (s + 1) * arm_radial + r
            p3 = base_idx_l + (s + 1) * arm_radial + next_r
            p4 = base_idx_l + s * arm_radial + next_r
            indices.extend([p1, p2, p3])
            indices.extend([p1, p3, p4])

    # Right Arm
    r_start = np.array([-1.4, 16.0, 0.3])
    r_end   = np.array([-6.8, 12.8, 0.25])
    base_idx_r = len(positions)
    for s in range(arm_steps):
        t = s / (arm_steps - 1)
        center = r_start + t * (r_end - r_start)
        v = 0.5 + t * 0.4
        for r in range(arm_radial):
            theta = 2.0 * np.pi * r / arm_radial
            u = r / float(arm_radial)
            p = center + np.array([0, r_arm * np.sin(theta), r_arm * np.cos(theta)])
            positions.append(list(p))
            normals.append([0, np.sin(theta), np.cos(theta)])
            uvs.append([u, v])
            joints.append([42, 43, 3, 0])
            weights.append([1.0 - t * 0.8, t * 0.8, 0.0, 0.0])

    for s in range(arm_steps - 1):
        for r in range(arm_radial):
            next_r = (r + 1) % arm_radial
            p1 = base_idx_r + s * arm_radial + r
            p2 = base_idx_r + (s + 1) * arm_radial + r
            p3 = base_idx_r + (s + 1) * arm_radial + next_r
            p4 = base_idx_r + s * arm_radial + next_r
            indices.extend([p1, p2, p3])
            indices.extend([p1, p3, p4])

    pos_np = np.array(positions, dtype=np.float32)
    nor_np = np.array(normals, dtype=np.float32)
    uvs_np = np.array(uvs, dtype=np.float32)
    idx_np = np.array(indices, dtype=np.uint32)
    jnt_np = np.array(joints, dtype=np.uint16)
    wgt_np = np.array(weights, dtype=np.float32)

    print(f"Total Bikini Vertices: {len(pos_np)}, Triangles: {len(idx_np)//3}")

    def append_data(data_bytes):
        if len(miku_bin) % 4 != 0:
            miku_bin.extend(b'\x00' * (4 - (len(miku_bin) % 4)))
        off = len(miku_bin)
        miku_bin.extend(data_bytes)
        return off

    off_pos = append_data(pos_np.tobytes())
    off_nor = append_data(nor_np.tobytes())
    off_uvs = append_data(uvs_np.tobytes())
    off_idx = append_data(idx_np.tobytes())
    off_jnt = append_data(jnt_np.tobytes())
    off_wgt = append_data(wgt_np.tobytes())

    def add_bv(off, byte_len, target=None):
        bv_idx = len(gltf['bufferViews'])
        d = {'buffer': 0, 'byteOffset': off, 'byteLength': byte_len}
        if target: d['target'] = target
        gltf['bufferViews'].append(d)
        return bv_idx

    bv_p = add_bv(off_pos, len(pos_np.tobytes()), 34962) # ARRAY_BUFFER
    bv_n = add_bv(off_nor, len(nor_np.tobytes()), 34962)
    bv_u = add_bv(off_uvs, len(uvs_np.tobytes()), 34962)
    bv_i = add_bv(off_idx, len(idx_np.tobytes()), 34963) # ELEMENT_ARRAY_BUFFER
    bv_j = add_bv(off_jnt, len(jnt_np.tobytes()), 34962)
    bv_w = add_bv(off_wgt, len(wgt_np.tobytes()), 34962)

    def add_acc(bv, comp_type, count, acc_type, min_val=None, max_val=None):
        acc_idx = len(gltf['accessors'])
        d = {'bufferView': bv, 'componentType': comp_type, 'count': count, 'type': acc_type}
        if min_val: d['min'] = min_val
        if max_val: d['max'] = max_val
        gltf['accessors'].append(d)
        return acc_idx

    acc_p = add_acc(bv_p, 5126, len(pos_np), 'VEC3', pos_np.min(axis=0).tolist(), pos_np.max(axis=0).tolist())
    acc_n = add_acc(bv_n, 5126, len(nor_np), 'VEC3')
    acc_u = add_acc(bv_u, 5126, len(uvs_np), 'VEC2')
    acc_i = add_acc(bv_i, 5125, len(idx_np), 'SCALAR', [int(idx_np.min())], [int(idx_np.max())])
    acc_j = add_acc(bv_j, 5123, len(jnt_np), 'VEC4') # UNSIGNED_SHORT
    acc_w = add_acc(bv_w, 5126, len(wgt_np), 'VEC4') # FLOAT

    bikini_mesh_idx = len(gltf['meshes'])
    gltf['meshes'].append({
        'name': 'Body_Miku_Bikini',
        'primitives': [{
            'attributes': {
                'POSITION': acc_p,
                'NORMAL': acc_n,
                'TEXCOORD_0': acc_u,
                'JOINTS_0': acc_j,
                'WEIGHTS_0': acc_w
            },
            'indices': acc_i,
            'material': mat_idx
        }]
    })

    bikini_node_idx = len(gltf['nodes'])
    gltf['nodes'].append({
        'name': 'Body_Miku_Bikini',
        'mesh': bikini_mesh_idx,
        'skin': 0
    })

    gltf['scenes'][0]['nodes'].append(bikini_node_idx)

    new_json_bytes = json.dumps(gltf, separators=(',', ':')).encode('utf-8')
    pad_json = (4 - (len(new_json_bytes) % 4)) % 4
    new_json_bytes += b' ' * pad_json

    pad_bin = (4 - (len(miku_bin) % 4)) % 4
    miku_bin += b'\x00' * pad_bin

    total_len = 12 + 8 + len(new_json_bytes) + 8 + len(miku_bin)

    with open(dst_glb, 'wb') as f:
        f.write(struct.pack('<4sII', magic, version, total_len))
        f.write(struct.pack('<I4s', len(new_json_bytes), chunk_type))
        f.write(new_json_bytes)
        f.write(struct.pack('<I4s', len(miku_bin), bin_type))
        f.write(miku_bin)

    size_mb = len(miku_bin) / (1024 * 1024)
    print(f"Successfully injected Body_Miku_Bikini into {dst_glb}! Buffer size: {size_mb:.2f} MB")

if __name__ == '__main__':
    build_miku_bikini()
