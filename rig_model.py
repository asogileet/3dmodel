import struct
import json
import os
import time
import numpy as np

def segment_distance(points, A, B):
    AB = B - A
    AB_len_sq = np.dot(AB, AB)
    if AB_len_sq < 1e-8:
        return np.linalg.norm(points - A, axis=1)
    AP = points - A
    t = np.clip(np.dot(AP, AB) / AB_len_sq, 0.0, 1.0)
    projection = A + t[:, np.newaxis] * AB
    return np.linalg.norm(points - projection, axis=1)

def main():
    start_time = time.time()
    input_glb = 'gloria.glb'
    output_glb = 'gloria_rigged.glb'
    
    print(f"Reading {input_glb}...")
    with open(input_glb, 'rb') as f:
        magic, version, length = struct.unpack('<4sII', f.read(12))
        chunk_len, chunk_type = struct.unpack('<I4s', f.read(8))
        json_bytes = f.read(chunk_len)
        gltf = json.loads(json_bytes.decode('utf-8'))
        bin_len, bin_type = struct.unpack('<I4s', f.read(8))
        bin_data = bytearray(f.read(bin_len))

    while len(bin_data) % 4 != 0:
        bin_data.append(0)

    # 17 bones
    bones_def = [
        {"name": "Hips",          "pos": np.array([ 0.000, 0.460,  0.010]), "parent": None, "end": np.array([ 0.000, 0.560,  0.010]), "type": "body"},
        {"name": "Spine",         "pos": np.array([ 0.000, 0.580,  0.010]), "parent": 0,    "end": np.array([ 0.000, 0.700,  0.015]), "type": "body"},
        {"name": "Chest",         "pos": np.array([ 0.000, 0.700,  0.015]), "parent": 1,    "end": np.array([ 0.000, 0.810, -0.015]), "type": "body"},
        {"name": "Neck",          "pos": np.array([ 0.000, 0.820, -0.020]), "parent": 2,    "end": np.array([ 0.000, 0.880,  0.000]), "type": "head"},
        {"name": "Head",          "pos": np.array([ 0.000, 0.880,  0.005]), "parent": 3,    "end": np.array([ 0.000, 0.980,  0.010]), "type": "head"},

        # Left Arm (User's right, positive X)
        {"name": "LeftUpperArm",  "pos": np.array([ 0.060, 0.700,  0.010]), "parent": 2,    "end": np.array([ 0.105, 0.560,  0.010]), "type": "arm_L"},
        {"name": "LeftLowerArm",  "pos": np.array([ 0.105, 0.560,  0.010]), "parent": 5,    "end": np.array([ 0.125, 0.420,  0.010]), "type": "arm_L"},
        {"name": "LeftHand",      "pos": np.array([ 0.125, 0.420,  0.010]), "parent": 6,    "end": np.array([ 0.138, 0.350,  0.010]), "type": "arm_L"},

        # Right Arm (User's left, negative X)
        {"name": "RightUpperArm", "pos": np.array([-0.060, 0.700,  0.010]), "parent": 2,    "end": np.array([-0.105, 0.560,  0.010]), "type": "arm_R"},
        {"name": "RightLowerArm", "pos": np.array([-0.105, 0.560,  0.010]), "parent": 8,    "end": np.array([-0.125, 0.420,  0.010]), "type": "arm_R"},
        {"name": "RightHand",     "pos": np.array([-0.125, 0.420,  0.010]), "parent": 9,    "end": np.array([-0.138, 0.350,  0.010]), "type": "arm_R"},

        # Left Leg (positive X)
        {"name": "LeftUpperLeg",  "pos": np.array([ 0.040, 0.440,  0.000]), "parent": 0,    "end": np.array([ 0.040, 0.230, -0.015]), "type": "leg_L"},
        {"name": "LeftLowerLeg",  "pos": np.array([ 0.040, 0.230, -0.015]), "parent": 11,   "end": np.array([ 0.030, 0.060, -0.025]), "type": "leg_L"},
        {"name": "LeftFoot",      "pos": np.array([ 0.030, 0.060, -0.025]), "parent": 12,   "end": np.array([ 0.025, 0.000,  0.020]), "type": "leg_L"},

        # Right Leg (negative X)
        {"name": "RightUpperLeg", "pos": np.array([-0.040, 0.440,  0.000]), "parent": 0,    "end": np.array([-0.040, 0.230, -0.015]), "type": "leg_R"},
        {"name": "RightLowerLeg", "pos": np.array([-0.040, 0.230, -0.015]), "parent": 14,   "end": np.array([-0.030, 0.060, -0.025]), "type": "leg_R"},
        {"name": "RightFoot",     "pos": np.array([-0.030, 0.060, -0.025]), "parent": 15,   "end": np.array([-0.025, 0.000,  0.020]), "type": "leg_R"},
    ]

    num_bones = len(bones_def)

    # Inverse bind matrices
    inv_bind_matrices = np.zeros((num_bones, 4, 4), dtype=np.float32)
    for i, b in enumerate(bones_def):
        mat = np.eye(4, dtype=np.float32)
        mat[3, :3] = -b['pos']
        inv_bind_matrices[i] = mat

    inv_bind_bytes = inv_bind_matrices.tobytes()
    ibm_bv_offset = len(bin_data)
    bin_data.extend(inv_bind_bytes)
    while len(bin_data) % 4 != 0:
        bin_data.append(0)

    ibm_bv_idx = len(gltf['bufferViews'])
    gltf['bufferViews'].append({
        "buffer": 0,
        "byteOffset": ibm_bv_offset,
        "byteLength": len(inv_bind_bytes)
    })

    ibm_acc_idx = len(gltf['accessors'])
    gltf['accessors'].append({
        "bufferView": ibm_bv_idx,
        "byteOffset": 0,
        "componentType": 5126,
        "count": num_bones,
        "type": "MAT4"
    })

    # Create Bone Nodes
    bone_start_node_idx = len(gltf['nodes'])
    bone_node_indices = []

    for i, b in enumerate(bones_def):
        curr_node_idx = bone_start_node_idx + i
        bone_node_indices.append(curr_node_idx)
        if b['parent'] is None:
            local_trans = b['pos'].tolist()
        else:
            parent_pos = bones_def[b['parent']]['pos']
            local_trans = (b['pos'] - parent_pos).tolist()

        gltf['nodes'].append({
            "name": f"Bone_{b['name']}",
            "translation": [round(x, 6) for x in local_trans],
            "children": []
        })

    for i, b in enumerate(bones_def):
        if b['parent'] is not None:
            parent_node_idx = bone_node_indices[b['parent']]
            gltf['nodes'][parent_node_idx]['children'].append(bone_node_indices[i])

    for idx in bone_node_indices:
        if not gltf['nodes'][idx]['children']:
            del gltf['nodes'][idx]['children']

    # Attach to Scene Root Node 3
    gltf['nodes'][3]['children'].append(bone_node_indices[0])

    # Create Skin
    skin_idx = len(gltf.get('skins', []))
    if 'skins' not in gltf:
        gltf['skins'] = []
    gltf['skins'].append({
        "name": "GloriaArmature",
        "inverseBindMatrices": ibm_acc_idx,
        "joints": bone_node_indices,
        "skeleton": bone_node_indices[0]
    })

    for node_idx in range(4, 22):
        gltf['nodes'][node_idx]['skin'] = skin_idx

    # Bone segment definitions
    A_list = [b['pos'] for b in bones_def]
    B_list = [b['end'] for b in bones_def]

    # Arm and Body bone index sets
    left_arm_indices  = {5, 6, 7}
    right_arm_indices = {8, 9, 10}
    arm_indices       = left_arm_indices | right_arm_indices
    body_leg_indices  = {0, 1, 2, 11, 12, 13, 14, 15, 16}
    head_indices      = {3, 4}

    total_prims = sum(len(m['primitives']) for m in gltf['meshes'])
    prim_count = 0

    print("Computing strict separated skinning weights...")

    for m_idx, mesh in enumerate(gltf['meshes']):
        for p_idx, prim in enumerate(mesh['primitives']):
            prim_count += 1
            pos_acc = gltf['accessors'][prim['attributes']['POSITION']]
            bv = gltf['bufferViews'][pos_acc['bufferView']]
            offset = bv.get('byteOffset', 0) + pos_acc.get('byteOffset', 0)
            count = pos_acc['count']

            raw_pos = bin_data[offset : offset + count * 12]
            verts = np.frombuffer(raw_pos, dtype=np.float32).reshape(-1, 3)

            # Compute raw distances to all bones
            dist_matrix = np.zeros((count, num_bones), dtype=np.float32)
            for b_i in range(num_bones):
                dist_matrix[:, b_i] = segment_distance(verts, A_list[b_i], B_list[b_i])

            # -------------------------------------------------------------
            # EXACT TOPOLOGICAL & SPATIAL ISOLATION
            # -------------------------------------------------------------
            # Below shoulder/armpit level (Y < 0.60), arms hang completely free
            # with a clear physical gap between |X| > 0.079 and |X| <= 0.079.
            y_coords = verts[:, 1]
            x_coords = verts[:, 0]

            is_lower_left_arm  = (y_coords < 0.60) & (x_coords > 0.079)
            is_lower_right_arm = (y_coords < 0.60) & (x_coords < -0.079)
            is_lower_body      = (y_coords < 0.60) & (np.abs(x_coords) <= 0.079)

            # 1. Lower Left Arm/Hand: ZERO weight from body, legs, head, and right arm
            for b_i in (body_leg_indices | head_indices | right_arm_indices):
                dist_matrix[is_lower_left_arm, b_i] += 1e6

            # 2. Lower Right Arm/Hand: ZERO weight from body, legs, head, and left arm
            for b_i in (body_leg_indices | head_indices | left_arm_indices):
                dist_matrix[is_lower_right_arm, b_i] += 1e6

            # 3. Lower Body/Hips/Legs: ZERO weight from ALL arm/hand bones
            for b_i in arm_indices:
                dist_matrix[is_lower_body, b_i] += 1e6

            # 4. Strict Left vs Right limb isolation across the entire body:
            for b_i in (left_arm_indices | {11, 12, 13}):
                dist_matrix[x_coords < -0.005, b_i] += 1e6
            for b_i in (right_arm_indices | {14, 15, 16}):
                dist_matrix[x_coords >  0.005, b_i] += 1e6

            # 5. Forearm & Hand should NEVER affect chest or spine
            for b_i in {6, 7, 9, 10}:
                dist_matrix[y_coords > 0.62, b_i] += 1e6

            # 6. Legs should NEVER affect upper torso or head (Y > 0.52)
            for b_i in {11, 12, 13, 14, 15, 16}:
                dist_matrix[y_coords > 0.52, b_i] += 1e6

            # 7. Head/Neck should NEVER affect below chest (Y < 0.75)
            for b_i in head_indices:
                dist_matrix[y_coords < 0.75, b_i] += 1e6

            # Convert distance to weight
            # High exponent gives crisp, clean deformation without rubbery bleeding
            inv_d = 1.0 / (dist_matrix + 0.005) ** 3.0

            # Find top 4 bones per vertex
            top4_indices = np.argsort(-inv_d, axis=1)[:, :4]
            rows = np.arange(count)[:, np.newaxis]
            top4_weights = inv_d[rows, top4_indices]

            # Normalize weights
            sum_weights = top4_weights.sum(axis=1, keepdims=True)
            sum_weights[sum_weights < 1e-8] = 1.0
            top4_weights = (top4_weights / sum_weights).astype(np.float32)

            top4_joints = top4_indices.astype(np.uint8)

            # Append JOINTS_0
            joints_bytes = top4_joints.tobytes()
            j_bv_offset = len(bin_data)
            bin_data.extend(joints_bytes)
            while len(bin_data) % 4 != 0:
                bin_data.append(0)

            j_bv_idx = len(gltf['bufferViews'])
            gltf['bufferViews'].append({
                "buffer": 0,
                "byteOffset": j_bv_offset,
                "byteLength": len(joints_bytes)
            })

            j_acc_idx = len(gltf['accessors'])
            gltf['accessors'].append({
                "bufferView": j_bv_idx,
                "byteOffset": 0,
                "componentType": 5121,
                "count": count,
                "type": "VEC4"
            })
            prim['attributes']['JOINTS_0'] = j_acc_idx

            # Append WEIGHTS_0
            weights_bytes = top4_weights.tobytes()
            w_bv_offset = len(bin_data)
            bin_data.extend(weights_bytes)
            while len(bin_data) % 4 != 0:
                bin_data.append(0)

            w_bv_idx = len(gltf['bufferViews'])
            gltf['bufferViews'].append({
                "buffer": 0,
                "byteOffset": w_bv_offset,
                "byteLength": len(weights_bytes)
            })

            w_acc_idx = len(gltf['accessors'])
            gltf['accessors'].append({
                "bufferView": w_bv_idx,
                "byteOffset": 0,
                "componentType": 5126,
                "count": count,
                "type": "VEC4"
            })
            prim['attributes']['WEIGHTS_0'] = w_acc_idx

            if prim_count % 3 == 0 or prim_count == total_prims:
                print(f"  Processed {prim_count}/{total_prims} primitives...")

    gltf['buffers'][0]['byteLength'] = len(bin_data)

    print("Serializing separated gloria_rigged.glb...")
    new_json_bytes = json.dumps(gltf, separators=(',', ':')).encode('utf-8')
    while len(new_json_bytes) % 4 != 0:
        new_json_bytes += b' '

    new_json_len = len(new_json_bytes)
    new_bin_len = len(bin_data)
    total_file_size = 12 + 8 + new_json_len + 8 + new_bin_len

    with open(output_glb, 'wb') as f:
        f.write(struct.pack('<4sII', b'glTF', 2, total_file_size))
        f.write(struct.pack('<I4s', new_json_len, b'JSON'))
        f.write(new_json_bytes)
        f.write(struct.pack('<I4s', new_bin_len, b'BIN\x00'))
        f.write(bin_data)

    elapsed = time.time() - start_time
    file_size_mb = os.path.getsize(output_glb) / (1024 * 1024)
    print(f"Done! Exported separated {output_glb} ({file_size_mb:.2f} MB) in {elapsed:.1f}s!")

if __name__ == '__main__':
    main()
