import bpy
import mathutils
import numpy as np
import math
import os
import sys

def segment_distance_batch(P, A, B):
    """P: (N, 3), A: (3,), B: (3,)"""
    AB = B - A
    AB_len_sq = np.dot(AB, AB)
    if AB_len_sq < 1e-8:
        return np.linalg.norm(P - A, axis=1)
    AP = P - A
    t = np.clip(np.dot(AP, AB) / AB_len_sq, 0.0, 1.0)
    projection = A + t[:, np.newaxis] * AB
    return np.linalg.norm(P - projection, axis=1)

def create_armature(bones_def):
    arm_data = bpy.data.armatures.new('CharacterArmature')
    arm_obj = bpy.data.objects.new('CharacterArmature', arm_data)
    bpy.context.scene.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj
    
    bpy.ops.object.mode_set(mode='EDIT')
    edit_bones = {}
    for b in bones_def:
        eb = arm_data.edit_bones.new(b['name'])
        eb.head = mathutils.Vector(b['head'])
        eb.tail = mathutils.Vector(b['tail'])
        edit_bones[b['name']] = eb
    
    for b in bones_def:
        if b['parent']:
            edit_bones[b['name']].parent = edit_bones[b['parent']]
    
    bpy.ops.object.mode_set(mode='OBJECT')
    return arm_obj

def apply_skinning_weights(mesh_obj, arm_obj, bones_def, is_tpose=True):
    verts = np.array([mesh_obj.matrix_world @ v.co for v in mesh_obj.data.vertices])
    N = len(verts)
    num_bones = len(bones_def)
    bone_names = [b['name'] for b in bones_def]
    
    # Distance matrix
    dist_matrix = np.zeros((N, num_bones), dtype=np.float32)
    for i, b in enumerate(bones_def):
        dist_matrix[:, i] = segment_distance_batch(verts, np.array(b['head']), np.array(b['tail']))
    
    # Bone index lookup
    b_idx = {b['name']: i for i, b in enumerate(bones_def)}
    left_arm_idx = {b_idx['LeftUpperArm'], b_idx['LeftLowerArm'], b_idx['LeftHand']}
    right_arm_idx = {b_idx['RightUpperArm'], b_idx['RightLowerArm'], b_idx['RightHand']}
    all_arm_idx = left_arm_idx | right_arm_idx
    left_leg_idx = {b_idx['LeftUpperLeg'], b_idx['LeftLowerLeg'], b_idx['LeftFoot']}
    right_leg_idx = {b_idx['RightUpperLeg'], b_idx['RightLowerLeg'], b_idx['RightFoot']}
    all_leg_idx = left_leg_idx | right_leg_idx
    head_idx = {b_idx['Neck'], b_idx['Head']}
    torso_idx = {b_idx['Hips'], b_idx['Spine'], b_idx['Chest']}
    
    x = verts[:, 0]
    z = verts[:, 2] # in Blender, Z is vertical!
    
    # 1. Left vs Right isolation
    for bi in (left_arm_idx | left_leg_idx):
        dist_matrix[x < -0.01, bi] += 1e6
    for bi in (right_arm_idx | right_leg_idx):
        dist_matrix[x > 0.01, bi] += 1e6
        
    # 2. Leg isolation: Legs never affect upper torso (Z > 0.3)
    for bi in all_leg_idx:
        dist_matrix[z > 0.15, bi] += 1e6
        
    # 3. Head isolation: Head/neck never affect lower body (Z < 0.4)
    for bi in head_idx:
        dist_matrix[z < 0.45, bi] += 1e6
        
    # 4. Arms isolation
    if is_tpose:
        # T-Pose: arms are out at X > 0.35, Z in [0.4, 0.7]
        # Hands/forearms never affect torso
        for bi in {b_idx['LeftLowerArm'], b_idx['LeftHand'], b_idx['RightLowerArm'], b_idx['RightHand']}:
            dist_matrix[np.abs(x) < 0.35, bi] += 1e6
        # Torso never affects outer arms
        for bi in (torso_idx | all_leg_idx):
            dist_matrix[np.abs(x) > 0.30, bi] += 1e6
    else:
        # A-Pose: arms hang down alongside hips/torso
        # Physical boundary: X > 0.20 and Z in [-0.15, 0.45]
        is_left_arm_zone = (x > 0.20) & (z < 0.45)
        is_right_arm_zone = (x < -0.20) & (z < 0.45)
        is_torso_leg_zone = (np.abs(x) <= 0.20) & (z < 0.45)
        
        for bi in (torso_idx | all_leg_idx | head_idx | right_arm_idx):
            dist_matrix[is_left_arm_zone, bi] += 1e6
        for bi in (torso_idx | all_leg_idx | head_idx | left_arm_idx):
            dist_matrix[is_right_arm_zone, bi] += 1e6
        for bi in all_arm_idx:
            dist_matrix[is_torso_leg_zone, bi] += 1e6
            
    # Softmax / power weighting
    inv_d = 1.0 / (dist_matrix + 0.005) ** 3.0
    top4 = np.argsort(-inv_d, axis=1)[:, :4]
    
    rows = np.arange(N)[:, np.newaxis]
    w4 = inv_d[rows, top4]
    w4_sum = w4.sum(axis=1, keepdims=True)
    w4_sum[w4_sum < 1e-8] = 1.0
    w4 = w4 / w4_sum
    
    # Create Vertex Groups in Blender
    mesh_obj.vertex_groups.clear()
    vgs = {name: mesh_obj.vertex_groups.new(name=name) for name in bone_names}
    
    for bone_col in range(4):
        indices = top4[:, bone_col]
        weights = w4[:, bone_col]
        for bi, bname in enumerate(bone_names):
            mask = (indices == bi) & (weights > 0.001)
            v_matches = np.where(mask)[0]
            if len(v_matches) > 0:
                for vi in v_matches:
                    vgs[bname].add([int(vi)], float(weights[vi]), 'ADD')
    
    # Add Armature modifier
    mod = mesh_obj.modifiers.new(name='Armature', type='ARMATURE')
    mod.object = arm_obj
    mesh_obj.parent = arm_obj

def add_actions(arm_obj):
    arm_obj.animation_data_create()
    
    # --- 1. IDLE (60 frames, 2.0s) ---
    idle_act = bpy.data.actions.new(name='Idle')
    arm_obj.animation_data.action = idle_act
    
    def insert_rot(bone_name, frame, rx, ry, rz):
        pb = arm_obj.pose.bones.get(bone_name)
        if not pb: return
        pb.rotation_mode = 'XYZ'
        pb.rotation_euler = (rx, ry, rz)
        pb.keyframe_insert(data_path='rotation_euler', frame=frame)
    
    # Idle breathing
    for f in [1, 60]:
        insert_rot('Chest', f, 0, 0, 0)
        insert_rot('Spine', f, 0, 0, 0)
        insert_rot('Head',  f, 0, 0, 0)
    insert_rot('Chest', 30, 0.06, 0, 0)
    insert_rot('Spine', 30, 0.03, 0, 0)
    insert_rot('Head',  30, -0.04, 0, 0)
    
    # --- 2. WALK (40 frames, 1.33s) ---
    walk_act = bpy.data.actions.new(name='Walk')
    arm_obj.animation_data.action = walk_act
    
    # Left / Right legs & arms alternating
    for f, (l_leg, r_leg, l_knee, r_knee, l_arm, r_arm) in [
        (1,  ( 0.45, -0.45,  0.0,   0.6, -0.35,  0.35)),
        (10, ( 0.00,  0.00,  0.2,   0.2,  0.00,  0.00)),
        (20, (-0.45,  0.45,  0.6,   0.0,  0.35, -0.35)),
        (30, ( 0.00,  0.00,  0.2,   0.2,  0.00,  0.00)),
        (40, ( 0.45, -0.45,  0.0,   0.6, -0.35,  0.35)),
    ]:
        insert_rot('LeftUpperLeg',  f, l_leg, 0, 0)
        insert_rot('RightUpperLeg', f, r_leg, 0, 0)
        insert_rot('LeftLowerLeg',  f, -l_knee, 0, 0)
        insert_rot('RightLowerLeg', f, -r_knee, 0, 0)
        insert_rot('LeftUpperArm',  f, l_arm, 0, 0)
        insert_rot('RightUpperArm', f, r_arm, 0, 0)
        
    # --- 3. DANCE (60 frames) ---
    dance_act = bpy.data.actions.new(name='Dance')
    arm_obj.animation_data.action = dance_act
    
    for f, (hip_y, hip_x, arm_l, arm_r) in [
        (1,  ( 0.15,  0.05,  0.3, -0.5)),
        (15, ( 0.00, -0.05, -0.2,  0.2)),
        (30, (-0.15,  0.05, -0.5,  0.3)),
        (45, ( 0.00, -0.05,  0.2, -0.2)),
        (60, ( 0.15,  0.05,  0.3, -0.5)),
    ]:
        insert_rot('Hips', f, hip_x, hip_y, 0)
        insert_rot('LeftUpperArm', f, arm_l, 0, 0.4)
        insert_rot('RightUpperArm', f, arm_r, 0, -0.4)
        
    # --- 4. WAVE (40 frames - natural greeting wave high up beside head) ---
    wave_act = bpy.data.actions.new(name='Wave')
    arm_obj.animation_data.action = wave_act
    
    insert_rot('RightUpperArm', 1, 0.5, 0.0, -1.2)
    insert_rot('RightUpperArm', 40, 0.5, 0.0, -1.2)
    for f, r_val in [(1, 1.2), (10, 1.6), (20, 1.2), (30, 1.6), (40, 1.2)]:
        insert_rot('RightLowerArm', f, r_val, 0.0, 0.0)
    insert_rot('Head', 1, 0.05, -0.12, 0.10)
    insert_rot('Head', 40, 0.05, -0.12, 0.10)

    # --- 5. SALUTE (40 frames - right hand to right eyebrow/temple) ---
    salute_act = bpy.data.actions.new(name='Salute')
    arm_obj.animation_data.action = salute_act
    for f in [1, 40]:
        insert_rot('RightUpperArm', f, 0.0, 0.5, -0.3)
        insert_rot('RightLowerArm', f, 2.3, 0.0, 0.0)
        insert_rot('Chest', f, 0.05, 0.0, 0.0)
        insert_rot('Head',  f, -0.04, 0.0, 0.0)
        
    # Set Idle as active action
    arm_obj.animation_data.action = idle_act

def rig_model(input_path, output_path, bones_def, is_tpose=True):
    print(f"\n==========================================")
    print(f"Rigging {os.path.basename(input_path)} -> {os.path.basename(output_path)}")
    print(f"==========================================")
    
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=input_path)
    
    meshes = [o for o in bpy.data.objects if o.type == 'MESH']
    if not meshes:
        raise Exception(f"No mesh found in {input_path}")
        
    mesh_obj = meshes[0]
    print(f"Imported mesh: {mesh_obj.name} with {len(mesh_obj.data.vertices)} vertices.")
    
    arm_obj = create_armature(bones_def)
    print(f"Created Armature with {len(bones_def)} bones.")
    
    print("Computing and applying skinning weights...")
    apply_skinning_weights(mesh_obj, arm_obj, bones_def, is_tpose)
    print("Skinning weights applied successfully!")
    
    print("Adding embedded actions (Idle, Walk, Dance, Wave)...")
    add_actions(arm_obj)
    
    print(f"Exporting GLB to {output_path}...")
    bpy.ops.export_scene.gltf(
        filepath=output_path,
        export_format='GLB',
        export_skins=True,
        export_animations=True,
        export_animation_mode='ACTIONS'
    )
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"SUCCESS! Exported {output_path} ({size_mb:.2f} MB)")

def main():
    # Model 1: T-Pose Girl
    girl_input = 'z:/home/edwintai/github/3dmodel/assets/nude_girl_t_pose_0310125114_texture_fbx.glb'
    girl_output = 'z:/home/edwintai/github/3dmodel/girl_rigged.glb'
    
    girl_bones = [
        {"name": "Hips",          "head": (0.00,  0.02,  0.05), "tail": (0.00,  0.01,  0.22), "parent": None},
        {"name": "Spine",         "head": (0.00,  0.01,  0.22), "tail": (0.00, -0.02,  0.45), "parent": "Hips"},
        {"name": "Chest",         "head": (0.00, -0.02,  0.45), "tail": (0.00,  0.00,  0.65), "parent": "Spine"},
        {"name": "Neck",          "head": (0.00,  0.00,  0.65), "tail": (0.00,  0.01,  0.73), "parent": "Chest"},
        {"name": "Head",          "head": (0.00,  0.01,  0.73), "tail": (0.00,  0.01,  0.98), "parent": "Neck"},

        # Left Arm (T-Pose along +X)
        {"name": "LeftUpperArm",  "head": (0.18, -0.02,  0.58), "tail": (0.45, -0.02,  0.58), "parent": "Chest"},
        {"name": "LeftLowerArm",  "head": (0.45, -0.02,  0.58), "tail": (0.72, -0.02,  0.58), "parent": "LeftUpperArm"},
        {"name": "LeftHand",      "head": (0.72, -0.02,  0.58), "tail": (0.96, -0.02,  0.58), "parent": "LeftLowerArm"},

        # Right Arm (T-Pose along -X)
        {"name": "RightUpperArm", "head": (-0.18, -0.02, 0.58), "tail": (-0.45, -0.02, 0.58), "parent": "Chest"},
        {"name": "RightLowerArm", "head": (-0.45, -0.02, 0.58), "tail": (-0.72, -0.02, 0.58), "parent": "RightUpperArm"},
        {"name": "RightHand",     "head": (-0.72, -0.02, 0.58), "tail": (-0.96, -0.02, 0.58), "parent": "RightLowerArm"},

        # Left Leg (along -Z, X ≈ +0.17)
        {"name": "LeftUpperLeg",  "head": (0.17,  0.01,  0.02), "tail": (0.18,  0.03, -0.45), "parent": "Hips"},
        {"name": "LeftLowerLeg",  "head": (0.18,  0.03, -0.45), "tail": (0.18, -0.02, -0.85), "parent": "LeftUpperLeg"},
        {"name": "LeftFoot",      "head": (0.18, -0.02, -0.85), "tail": (0.18,  0.10, -0.98), "parent": "LeftLowerLeg"},

        # Right Leg (along -Z, X ≈ -0.17)
        {"name": "RightUpperLeg", "head": (-0.17,  0.01, 0.02), "tail": (-0.18,  0.03, -0.45), "parent": "Hips"},
        {"name": "RightLowerLeg", "head": (-0.18,  0.03, -0.45), "tail": (-0.18, -0.02, -0.85), "parent": "RightUpperLeg"},
        {"name": "RightFoot",     "head": (-0.18, -0.02, -0.85), "tail": (-0.18,  0.10, -0.98), "parent": "RightLowerLeg"},
    ]
    
    rig_model(girl_input, girl_output, girl_bones, is_tpose=True)
    
    # Model 2: A-Pose Woman
    woman_input = 'z:/home/edwintai/github/3dmodel/assets/adult-base-nude-woman/source/model.glb'
    woman_output = 'z:/home/edwintai/github/3dmodel/woman_rigged.glb'
    
    woman_bones = [
        {"name": "Hips",          "head": (0.00, -0.01,  0.02), "tail": (0.00,  0.00,  0.20), "parent": None},
        {"name": "Spine",         "head": (0.00,  0.00,  0.20), "tail": (0.00,  0.02,  0.42), "parent": "Hips"},
        {"name": "Chest",         "head": (0.00,  0.02,  0.42), "tail": (0.00, -0.02,  0.62), "parent": "Spine"},
        {"name": "Neck",          "head": (0.00, -0.02,  0.62), "tail": (0.00,  0.00,  0.70), "parent": "Chest"},
        {"name": "Head",          "head": (0.00,  0.00,  0.70), "tail": (0.00,  0.02,  0.98), "parent": "Neck"},

        # Left Arm (A-Pose along diagonal)
        {"name": "LeftUpperArm",  "head": (0.19, -0.05,  0.50), "tail": (0.26, -0.05,  0.32), "parent": "Chest"},
        {"name": "LeftLowerArm",  "head": (0.26, -0.05,  0.32), "tail": (0.32, -0.02,  0.12), "parent": "LeftUpperArm"},
        {"name": "LeftHand",      "head": (0.32, -0.02,  0.12), "tail": (0.36, -0.01, -0.08), "parent": "LeftLowerArm"},

        # Right Arm (A-Pose along diagonal)
        {"name": "RightUpperArm", "head": (-0.19, -0.05, 0.50), "tail": (-0.26, -0.05, 0.32), "parent": "Chest"},
        {"name": "RightLowerArm", "head": (-0.26, -0.05, 0.32), "tail": (-0.32, -0.02, 0.12), "parent": "RightUpperArm"},
        {"name": "RightHand",     "head": (-0.32, -0.02, 0.12), "tail": (-0.36, -0.01, -0.08), "parent": "RightLowerArm"},

        # Left Leg (along -Z, X ≈ +0.10)
        {"name": "LeftUpperLeg",  "head": (0.10, -0.02,  0.00), "tail": (0.09, -0.03, -0.45), "parent": "Hips"},
        {"name": "LeftLowerLeg",  "head": (0.09, -0.03, -0.45), "tail": (0.08, -0.06, -0.85), "parent": "LeftUpperLeg"},
        {"name": "LeftFoot",      "head": (0.08, -0.06, -0.85), "tail": (0.08,  0.06, -0.98), "parent": "LeftLowerLeg"},

        # Right Leg (along -Z, X ≈ -0.10)
        {"name": "RightUpperLeg", "head": (-0.10, -0.02, 0.00), "tail": (-0.09, -0.03, -0.45), "parent": "Hips"},
        {"name": "RightLowerLeg", "head": (-0.09, -0.03, -0.45), "tail": (-0.08, -0.06, -0.85), "parent": "RightUpperLeg"},
        {"name": "RightFoot",     "head": (-0.08, -0.06, -0.85), "tail": (-0.08,  0.06, -0.98), "parent": "RightLowerLeg"},
    ]
    
    rig_model(woman_input, woman_output, woman_bones, is_tpose=False)

if __name__ == '__main__':
    main()
