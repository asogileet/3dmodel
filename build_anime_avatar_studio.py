import bpy
import bmesh
import mathutils
import numpy as np
import math
import os
import sys

# -----------------------------------------------------------------------------
# Math & Skinning Utilities
# -----------------------------------------------------------------------------
def segment_distance_batch(P, A, B):
    AB = B - A
    AB_len_sq = np.dot(AB, AB)
    if AB_len_sq < 1e-8:
        return np.linalg.norm(P - A, axis=1)
    AP = P - A
    t = np.clip(np.dot(AP, AB) / AB_len_sq, 0.0, 1.0)
    proj = A + t[:, np.newaxis] * AB
    return np.linalg.norm(P - proj, axis=1)

def apply_weights_to_mesh(mesh_obj, arm_obj, bones_def, is_hair=False, is_head_only=False):
    verts = np.array([mesh_obj.matrix_world @ v.co for v in mesh_obj.data.vertices])
    N = len(verts)
    if N == 0: return

    bone_names = [b['name'] for b in bones_def]
    num_bones = len(bones_def)
    b_idx = {b['name']: i for i, b in enumerate(bones_def)}

    if is_head_only:
        mesh_obj.vertex_groups.clear()
        vg_head = mesh_obj.vertex_groups.new(name='Head')
        vg_head.add(list(range(N)), 1.0, 'REPLACE')
        mod = mesh_obj.modifiers.new(name='Armature', type='ARMATURE')
        mod.object = arm_obj
        mesh_obj.parent = arm_obj
        return

    dist_matrix = np.zeros((N, num_bones), dtype=np.float32)
    for i, b in enumerate(bones_def):
        dist_matrix[:, i] = segment_distance_batch(verts, np.array(b['head']), np.array(b['tail']))

    x = verts[:, 0]
    z = verts[:, 2]

    left_arm = {b_idx.get('LeftUpperArm'), b_idx.get('LeftLowerArm'), b_idx.get('LeftHand')} - {None}
    right_arm = {b_idx.get('RightUpperArm'), b_idx.get('RightLowerArm'), b_idx.get('RightHand')} - {None}
    left_leg = {b_idx.get('LeftUpperLeg'), b_idx.get('LeftLowerLeg'), b_idx.get('LeftFoot')} - {None}
    right_leg = {b_idx.get('RightUpperLeg'), b_idx.get('RightLowerLeg'), b_idx.get('RightFoot')} - {None}
    head_bones = {b_idx.get('Head'), b_idx.get('Neck')} - {None}
    torso_bones = {b_idx.get('Hips'), b_idx.get('Spine'), b_idx.get('Chest')} - {None}

    # Left / Right isolation
    for bi in (left_arm | left_leg):
        dist_matrix[x < -0.01, bi] += 1e6
    for bi in (right_arm | right_leg):
        dist_matrix[x > 0.01, bi] += 1e6

    # Legs never affect upper torso
    for bi in (left_leg | right_leg):
        dist_matrix[z > 0.15, bi] += 1e6

    # Head never affects lower body
    for bi in head_bones:
        dist_matrix[z < 0.50, bi] += 1e6

    # Arm isolation
    for bi in (left_arm | right_arm):
        dist_matrix[(np.abs(x) < 0.14) & (z < 0.50), bi] += 1e6

    # Twintails
    if is_hair:
        lt1 = b_idx.get('LeftTwintail_1')
        lt2 = b_idx.get('LeftTwintail_2')
        rt1 = b_idx.get('RightTwintail_1')
        rt2 = b_idx.get('RightTwintail_2')
        if lt1 is not None and lt2 is not None:
            dist_matrix[x < 0.05, lt1] += 1e6
            dist_matrix[x < 0.05, lt2] += 1e6
        if rt1 is not None and rt2 is not None:
            dist_matrix[x > -0.05, rt1] += 1e6
            dist_matrix[x > -0.05, rt2] += 1e6

    inv_d = 1.0 / (dist_matrix + 0.005) ** 3.0
    top4 = np.argsort(-inv_d, axis=1)[:, :4]
    rows = np.arange(N)[:, np.newaxis]
    w4 = inv_d[rows, top4]
    w4_sum = w4.sum(axis=1, keepdims=True)
    w4_sum[w4_sum < 1e-8] = 1.0
    w4 = w4 / w4_sum

    mesh_obj.vertex_groups.clear()
    vgs = {name: mesh_obj.vertex_groups.new(name=name) for name in bone_names}

    for col in range(4):
        indices = top4[:, col]
        weights = w4[:, col]
        for bi, bname in enumerate(bone_names):
            mask = (indices == bi) & (weights > 0.001)
            v_matches = np.where(mask)[0]
            if len(v_matches) > 0:
                for vi in v_matches:
                    vgs[bname].add([int(vi)], float(weights[vi]), 'ADD')

    mod = mesh_obj.modifiers.new(name='Armature', type='ARMATURE')
    mod.object = arm_obj
    mesh_obj.parent = arm_obj

# -----------------------------------------------------------------------------
# Material Creation
# -----------------------------------------------------------------------------
def create_anime_material(name, base_color, roughness=0.5, metallic=0.0):
    mat = bpy.data.materials.get(name)
    if not mat:
        mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs['Base Color'].default_value = base_color
        bsdf.inputs['Roughness'].default_value = roughness
        bsdf.inputs['Metallic'].default_value = metallic
    return mat

# -----------------------------------------------------------------------------
# Armature Creation
# -----------------------------------------------------------------------------
def build_armature():
    arm_data = bpy.data.armatures.new('AnimeArmature')
    arm_obj = bpy.data.objects.new('AnimeArmature', arm_data)
    bpy.context.scene.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj

    bones_def = [
        {"name": "Hips",          "head": (0.00,  0.00,  0.05), "tail": (0.00,  0.00,  0.22), "parent": None},
        {"name": "Spine",         "head": (0.00,  0.00,  0.22), "tail": (0.00, -0.01,  0.44), "parent": "Hips"},
        {"name": "Chest",         "head": (0.00, -0.01,  0.44), "tail": (0.00,  0.00,  0.64), "parent": "Spine"},
        {"name": "Neck",          "head": (0.00,  0.00,  0.64), "tail": (0.00,  0.01,  0.72), "parent": "Chest"},
        {"name": "Head",          "head": (0.00,  0.01,  0.72), "tail": (0.00,  0.01,  0.98), "parent": "Neck"},

        # Left Arm
        {"name": "LeftUpperArm",  "head": (0.18,  0.00,  0.58), "tail": (0.44,  0.00,  0.58), "parent": "Chest"},
        {"name": "LeftLowerArm",  "head": (0.44,  0.00,  0.58), "tail": (0.70,  0.00,  0.58), "parent": "LeftUpperArm"},
        {"name": "LeftHand",      "head": (0.70,  0.00,  0.58), "tail": (0.92,  0.00,  0.58), "parent": "LeftLowerArm"},

        # Right Arm
        {"name": "RightUpperArm", "head": (-0.18, 0.00,  0.58), "tail": (-0.44, 0.00,  0.58), "parent": "Chest"},
        {"name": "RightLowerArm", "head": (-0.44, 0.00,  0.58), "tail": (-0.70, 0.00,  0.58), "parent": "RightUpperArm"},
        {"name": "RightHand",     "head": (-0.70, 0.00,  0.58), "tail": (-0.92, 0.00,  0.58), "parent": "RightLowerArm"},

        # Left Leg
        {"name": "LeftUpperLeg",  "head": (0.14,  0.00,  0.02), "tail": (0.14,  0.01, -0.45), "parent": "Hips"},
        {"name": "LeftLowerLeg",  "head": (0.14,  0.01, -0.45), "tail": (0.14, -0.02, -0.84), "parent": "LeftUpperLeg"},
        {"name": "LeftFoot",      "head": (0.14, -0.02, -0.84), "tail": (0.14,  0.10, -0.98), "parent": "LeftLowerLeg"},

        # Right Leg
        {"name": "RightUpperLeg", "head": (-0.14, 0.00,  0.02), "tail": (-0.14, 0.01, -0.45), "parent": "Hips"},
        {"name": "RightLowerLeg", "head": (-0.14, 0.01, -0.45), "tail": (-0.14, -0.02, -0.84), "parent": "RightUpperLeg"},
        {"name": "RightFoot",     "head": (-0.14, -0.02, -0.84), "tail": (-0.14, 0.10, -0.98), "parent": "RightLowerLeg"},

        # Hair Twintail Bones
        {"name": "LeftTwintail_1",  "head": ( 0.22, -0.05, 0.88), "tail": ( 0.32, -0.08, 0.60), "parent": "Head"},
        {"name": "LeftTwintail_2",  "head": ( 0.32, -0.08, 0.60), "tail": ( 0.36, -0.10, 0.25), "parent": "LeftTwintail_1"},
        {"name": "RightTwintail_1", "head": (-0.22, -0.05, 0.88), "tail": (-0.32, -0.08, 0.60), "parent": "Head"},
        {"name": "RightTwintail_2", "head": (-0.32, -0.08, 0.60), "tail": (-0.36, -0.10, 0.25), "parent": "RightTwintail_1"},
    ]

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
    return arm_obj, bones_def

# -----------------------------------------------------------------------------
# Procedural Anime Geometry Builders
# -----------------------------------------------------------------------------
def add_cylinder_segment(bm, r_top, r_bot, z_top, z_bot, x_center=0.0, y_center=0.0, segs=12):
    top_verts = []
    bot_verts = []
    for i in range(segs):
        ang = i * 2.0 * math.pi / segs
        ca, sa = math.cos(ang), math.sin(ang)
        v_top = bm.verts.new((x_center + r_top * ca, y_center + r_top * sa, z_top))
        v_bot = bm.verts.new((x_center + r_bot * ca, y_center + r_bot * sa, z_bot))
        top_verts.append(v_top)
        bot_verts.append(v_bot)
    bm.verts.ensure_lookup_table()
    for i in range(segs):
        i_next = (i + 1) % segs
        bm.faces.new([top_verts[i], top_verts[i_next], bot_verts[i_next], bot_verts[i]])

def add_capsule_limb(bm, p_start, p_end, radius, segs=10):
    start = mathutils.Vector(p_start)
    end = mathutils.Vector(p_end)
    dir_v = end - start
    length = dir_v.length
    if length < 1e-4: return
    dir_n = dir_v.normalized()

    # Find perpendicular vectors
    up = mathutils.Vector((0, 0, 1)) if abs(dir_n.z) < 0.9 else mathutils.Vector((0, 1, 0))
    right = dir_n.cross(up).normalized()
    up = right.cross(dir_n).normalized()

    rings = 4
    rings_verts = []
    for r in range(rings):
        t = r / (rings - 1)
        center = start + dir_v * t
        ring = []
        for i in range(segs):
            ang = i * 2.0 * math.pi / segs
            offset = (right * math.cos(ang) + up * math.sin(ang)) * radius
            ring.append(bm.verts.new(center + offset))
        rings_verts.append(ring)

    for r in range(rings - 1):
        for i in range(segs):
            i_next = (i + 1) % segs
            v0 = rings_verts[r][i]
            v1 = rings_verts[r][i_next]
            v2 = rings_verts[r + 1][i_next]
            v3 = rings_verts[r + 1][i]
            bm.faces.new([v0, v1, v2, v3])

# 1. Anime Head Builder
def create_anime_head_bmesh(gender='female'):
    bm = bmesh.new()
    # Base anime skull
    bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=14, radius=0.17)
    
    # Stylize into cute Anime Teardrop Face
    scale_y = 1.05 if gender == 'female' else 1.02
    chin_point = 0.28 if gender == 'female' else 0.38
    jaw_width = 0.90 if gender == 'female' else 1.05

    for v in bm.verts:
        v.co.z += 0.85
        rel_z = v.co.z - 0.85

        # Pinch chin at bottom (rel_z < -0.04)
        if rel_z < -0.02:
            t = np.clip((-rel_z - 0.02) / 0.14, 0.0, 1.0)
            f = 1.0 - t * (1.0 - chin_point)
            v.co.x *= f * jaw_width
            v.co.y *= f
            # Cute anime slight forward chin point
            if rel_z < -0.08:
                v.co.y += (1.0 - t) * 0.02

        # Cheeks and forehead
        if rel_z > 0.0 and v.co.y > 0.0:
            v.co.y *= scale_y

    # Anime Ears
    for side in [1.0, -1.0]:
        ear_c = mathutils.Vector((side * 0.16, -0.02, 0.84))
        v1 = bm.verts.new(ear_c + mathutils.Vector((0, -0.02, 0.03)))
        v2 = bm.verts.new(ear_c + mathutils.Vector((side * 0.03, 0.01, 0.02)))
        v3 = bm.verts.new(ear_c + mathutils.Vector((side * 0.02, 0.02, -0.02)))
        v4 = bm.verts.new(ear_c + mathutils.Vector((0, -0.01, -0.03)))
        bm.faces.new([v1, v2, v3, v4])

    return bm

# 2. Anime Body Mesh Builder
def create_anime_body(name, gender='female'):
    bm = create_anime_head_bmesh(gender)

    shoulder_w = 0.18 if gender == 'female' else 0.23
    chest_depth = 0.12 if gender == 'female' else 0.11
    waist_w = 0.12 if gender == 'female' else 0.14
    hip_w = 0.16 if gender == 'female' else 0.15
    arm_r = 0.034 if gender == 'female' else 0.042
    leg_r_top = 0.065 if gender == 'female' else 0.070
    leg_r_bot = 0.035 if gender == 'female' else 0.040

    # Neck
    add_cylinder_segment(bm, 0.042, 0.048, 0.72, 0.64, 0.0, 0.01, segs=10)

    # Chest & Upper Torso
    add_cylinder_segment(bm, shoulder_w, waist_w, 0.64, 0.44, 0.0, 0.0, segs=14)

    # Anime Bust (Cute subtle anime breast volume for female)
    if gender == 'female':
        for side in [1.0, -1.0]:
            bust_c = mathutils.Vector((side * 0.065, 0.085, 0.53))
            v_b1 = bm.verts.new(bust_c + mathutils.Vector((0, 0.035, 0)))
            v_b2 = bm.verts.new(bust_c + mathutils.Vector((side * 0.04, -0.01, 0.03)))
            v_b3 = bm.verts.new(bust_c + mathutils.Vector((side * 0.03, -0.01, -0.03)))
            v_b4 = bm.verts.new(bust_c + mathutils.Vector((-side * 0.03, -0.01, -0.02)))
            v_b5 = bm.verts.new(bust_c + mathutils.Vector((-side * 0.03, -0.01, 0.02)))
            bm.faces.new([v_b1, v_b2, v_b3])
            bm.faces.new([v_b1, v_b3, v_b4])
            bm.faces.new([v_b1, v_b4, v_b5])
            bm.faces.new([v_b1, v_b5, v_b2])

    # Waist to Pelvis/Hips
    add_cylinder_segment(bm, waist_w, hip_w, 0.44, 0.22, 0.0, 0.0, segs=14)
    add_cylinder_segment(bm, hip_w, hip_w * 0.9, 0.22, 0.05, 0.0, 0.0, segs=14)

    # Arms (Left & Right)
    for side in [1.0, -1.0]:
        sh_pos = (side * shoulder_w, 0.0, 0.58)
        el_pos = (side * (shoulder_w + 0.26), 0.0, 0.58)
        wr_pos = (side * (shoulder_w + 0.52), 0.0, 0.58)
        hd_pos = (side * (shoulder_w + 0.68), 0.0, 0.58)

        add_capsule_limb(bm, sh_pos, el_pos, arm_r, segs=8)
        add_capsule_limb(bm, el_pos, wr_pos, arm_r * 0.85, segs=8)
        # Cute Anime Hand
        add_capsule_limb(bm, wr_pos, hd_pos, arm_r * 0.75, segs=6)

    # Legs (Left & Right)
    for side in [1.0, -1.0]:
        hip_pos = (side * (hip_w * 0.65), 0.0, 0.02)
        knee_pos = (side * (hip_w * 0.65), 0.01, -0.45)
        ankle_pos = (side * (hip_w * 0.65), -0.02, -0.84)
        toe_pos = (side * (hip_w * 0.65), 0.12, -0.98)

        add_capsule_limb(bm, hip_pos, knee_pos, leg_r_top, segs=10)
        add_capsule_limb(bm, knee_pos, ankle_pos, leg_r_bot, segs=10)
        # Anime Foot
        add_capsule_limb(bm, ankle_pos, toe_pos, leg_r_bot * 0.9, segs=8)

    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    mat_skin = create_anime_material('Mat_Skin', (0.98, 0.88, 0.82, 1.0), roughness=0.45)
    obj.data.materials.append(mat_skin)
    return obj

# 3. Anime Eye Decals (Curved Eye Planes for Live Expressions)
def create_anime_eyes(name):
    bm = bmesh.new()
    for side in [1.0, -1.0]:
        cx = side * 0.075
        cy = 0.148
        cz = 0.845
        w = 0.038
        h = 0.042

        # Curved 4-vertex quad matching anime face contour
        v1 = bm.verts.new((cx - side * w, cy - 0.008, cz + h))
        v2 = bm.verts.new((cx + side * w, cy + 0.002, cz + h))
        v3 = bm.verts.new((cx + side * w, cy + 0.002, cz - h))
        v4 = bm.verts.new((cx - side * w, cy - 0.008, cz - h))

        bm.faces.new([v1, v2, v3, v4])

    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    mat_eyes = create_anime_material('Mat_Eyes', (0.15, 0.45, 0.95, 1.0), roughness=0.1)
    obj.data.materials.append(mat_eyes)
    return obj

# 4. Hairstyles Builders
def create_hair_twintails(name):
    bm = bmesh.new()
    # Hair Cap
    bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=12, radius=0.185)
    for v in bm.verts:
        v.co.z += 0.87
        if v.co.z < 0.77: # trim bottom
            v.co.z = 0.77

    # Anime Bangs (Front fringe)
    for i in range(7):
        ang = (i - 3) * 0.18
        x = math.sin(ang) * 0.17
        y = 0.16 + math.cos(ang) * 0.02
        z_start = 0.93
        z_end = 0.85 - abs(i - 3) * 0.015
        add_capsule_limb(bm, (x, y, z_start), (x * 1.05, y + 0.01, z_end), 0.018, segs=6)

    # Left & Right Twintails
    for side in [1.0, -1.0]:
        base = mathutils.Vector((side * 0.22, -0.05, 0.88))
        mid = mathutils.Vector((side * 0.34, -0.08, 0.60))
        end = mathutils.Vector((side * 0.38, -0.10, 0.25))

        # Ribbon tie
        add_cylinder_segment(bm, 0.038, 0.038, 0.89, 0.86, side * 0.22, -0.05, segs=8)
        # Flowing locks
        add_capsule_limb(bm, base, mid, 0.045, segs=8)
        add_capsule_limb(bm, mid, end, 0.032, segs=8)

    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    mat_hair = create_anime_material('Mat_Hair', (0.20, 0.78, 0.88, 1.0), roughness=0.35)
    obj.data.materials.append(mat_hair)
    return obj

def create_hair_bob(name):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=14, radius=0.19)
    for v in bm.verts:
        v.co.z += 0.86
        # Cute rounded flare
        if 0.72 < v.co.z < 0.82:
            v.co.x *= 1.08
            v.co.y *= 1.08
        if v.co.z < 0.72:
            v.co.z = 0.72

    # Front bangs
    for i in range(6):
        x = (i - 2.5) * 0.045
        add_capsule_limb(bm, (x, 0.17, 0.94), (x, 0.18, 0.85), 0.02, segs=6)

    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    mat_hair = create_anime_material('Mat_Hair', (0.45, 0.25, 0.15, 1.0), roughness=0.35)
    obj.data.materials.append(mat_hair)
    return obj

def create_hair_hime(name):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=14, radius=0.188)
    for v in bm.verts:
        v.co.z += 0.87
        if v.co.z < 0.80 and v.co.y > 0.02:
            v.co.z = 0.80

    # Long straight back hair down to waist (Z = 0.35)
    for side in [-1, 0, 1]:
        x = side * 0.08
        add_capsule_limb(bm, (x, -0.14, 0.82), (x * 1.2, -0.12, 0.35), 0.04, segs=8)

    # Traditional Hime blunt cheek locks
    for side in [1.0, -1.0]:
        add_capsule_limb(bm, (side * 0.16, 0.05, 0.88), (side * 0.15, 0.05, 0.68), 0.025, segs=6)

    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    mat_hair = create_anime_material('Mat_Hair', (0.10, 0.10, 0.12, 1.0), roughness=0.35)
    obj.data.materials.append(mat_hair)
    return obj

def create_hair_spiky(name):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=12, radius=0.185)
    for v in bm.verts:
        v.co.z += 0.87
        if v.co.z < 0.78: v.co.z = 0.78

    # Anime Shonen Spikes
    spikes = [
        ((0.00, 0.05, 1.04), (0.00, 0.08, 1.14), 0.035),
        ((0.08, 0.02, 1.02), (0.14, 0.05, 1.12), 0.030),
        ((-0.08, 0.02, 1.02), (-0.14, 0.05, 1.12), 0.030),
        ((0.15, -0.05, 0.98), (0.24, -0.06, 1.04), 0.028),
        ((-0.15, -0.05, 0.98), (-0.24, -0.06, 1.04), 0.028),
        ((0.00, -0.15, 0.96), (0.00, -0.22, 1.02), 0.032),
        ((0.04, 0.16, 0.92), (0.05, 0.20, 0.84), 0.022),
        ((-0.04, 0.16, 0.92), (-0.05, 0.20, 0.84), 0.022),
    ]
    for p_base, p_tip, r in spikes:
        add_capsule_limb(bm, p_base, p_tip, r, segs=6)

    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    mat_hair = create_anime_material('Mat_Hair', (0.95, 0.75, 0.20, 1.0), roughness=0.35)
    obj.data.materials.append(mat_hair)
    return obj

def create_hair_parted(name):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=12, radius=0.186)
    for v in bm.verts:
        v.co.z += 0.87
        if v.co.z < 0.78: v.co.z = 0.78

    # Elegant side parted strands
    add_capsule_limb(bm, (-0.02, 0.16, 0.96), (0.12, 0.17, 0.86), 0.026, segs=6)
    add_capsule_limb(bm, (0.08, 0.15, 0.94), (0.16, 0.14, 0.82), 0.024, segs=6)
    add_capsule_limb(bm, (-0.06, 0.15, 0.95), (-0.14, 0.14, 0.86), 0.024, segs=6)

    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    mat_hair = create_anime_material('Mat_Hair', (0.25, 0.20, 0.35, 1.0), roughness=0.35)
    obj.data.materials.append(mat_hair)
    return obj

# 5. Outfits Builders
def create_outfit_sailor(name):
    bm = bmesh.new()
    # Sailor Top
    add_cylinder_segment(bm, 0.19, 0.13, 0.65, 0.42, 0.0, 0.0, segs=14)
    # Sailor Wide Back Collar
    v_c1 = bm.verts.new((-0.16, -0.16, 0.64))
    v_c2 = bm.verts.new(( 0.16, -0.16, 0.64))
    v_c3 = bm.verts.new(( 0.18,  0.02, 0.62))
    v_c4 = bm.verts.new((-0.18,  0.02, 0.62))
    bm.faces.new([v_c1, v_c2, v_c3, v_c4])

    # Red Bowknot / Ribbon
    v_r1 = bm.verts.new(( 0.00,  0.11, 0.58))
    v_r2 = bm.verts.new((-0.05,  0.12, 0.55))
    v_r3 = bm.verts.new(( 0.00,  0.11, 0.52))
    v_r4 = bm.verts.new(( 0.05,  0.12, 0.55))
    bm.faces.new([v_r1, v_r2, v_r3])
    bm.faces.new([v_r1, v_r3, v_r4])

    # Pleated Skirt (Z = 0.42 to 0.08)
    add_cylinder_segment(bm, 0.13, 0.24, 0.42, 0.08, 0.0, 0.0, segs=16)

    # Sleeves
    for side in [1.0, -1.0]:
        add_capsule_limb(bm, (side * 0.18, 0.0, 0.58), (side * 0.30, 0.0, 0.58), 0.046, segs=8)

    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    mat_outfit = create_anime_material('Mat_Outfit_Primary', (0.12, 0.18, 0.35, 1.0), roughness=0.6)
    obj.data.materials.append(mat_outfit)
    return obj

def create_outfit_blazer(name):
    bm = bmesh.new()
    # Blazer Jacket
    add_cylinder_segment(bm, 0.21, 0.15, 0.65, 0.35, 0.0, 0.0, segs=14)
    # Lapels
    for side in [1.0, -1.0]:
        add_capsule_limb(bm, (side * 0.04, 0.12, 0.63), (side * 0.08, 0.12, 0.48), 0.02, segs=6)
    # Tie
    add_capsule_limb(bm, (0.0, 0.10, 0.62), (0.0, 0.10, 0.44), 0.016, segs=6)
    # Trousers / Slacks from Z=0.35 to -0.80
    for side in [1.0, -1.0]:
        add_capsule_limb(bm, (side * 0.12, 0.0, 0.35), (side * 0.14, 0.01, -0.45), 0.075, segs=10)
        add_capsule_limb(bm, (side * 0.14, 0.01, -0.45), (side * 0.14, -0.02, -0.80), 0.055, segs=10)
    # Sleeves
    for side in [1.0, -1.0]:
        add_capsule_limb(bm, (side * 0.20, 0.0, 0.58), (side * 0.46, 0.0, 0.58), 0.048, segs=8)
        add_capsule_limb(bm, (side * 0.46, 0.0, 0.58), (side * 0.68, 0.0, 0.58), 0.042, segs=8)

    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    mat_outfit = create_anime_material('Mat_Outfit_Primary', (0.15, 0.15, 0.18, 1.0), roughness=0.6)
    obj.data.materials.append(mat_outfit)
    return obj

def create_outfit_hoodie(name):
    bm = bmesh.new()
    # Baggy Oversized Hoodie Body
    add_cylinder_segment(bm, 0.22, 0.18, 0.65, 0.25, 0.0, 0.0, segs=14)
    # Hood draped on back of neck
    add_cylinder_segment(bm, 0.18, 0.14, 0.72, 0.62, 0.0, -0.10, segs=10)
    # Kangaroo Pocket
    v_p1 = bm.verts.new((-0.10, 0.15, 0.38))
    v_p2 = bm.verts.new(( 0.10, 0.15, 0.38))
    v_p3 = bm.verts.new(( 0.12, 0.14, 0.27))
    v_p4 = bm.verts.new((-0.12, 0.14, 0.27))
    bm.faces.new([v_p1, v_p2, v_p3, v_p4])
    # Baggy Sleeves
    for side in [1.0, -1.0]:
        add_capsule_limb(bm, (side * 0.20, 0.0, 0.58), (side * 0.45, 0.0, 0.58), 0.055, segs=8)
        add_capsule_limb(bm, (side * 0.45, 0.0, 0.58), (side * 0.68, 0.0, 0.58), 0.046, segs=8)
    # Shorts
    for side in [1.0, -1.0]:
        add_capsule_limb(bm, (side * 0.12, 0.0, 0.25), (side * 0.14, 0.0, 0.02), 0.078, segs=10)

    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    mat_outfit = create_anime_material('Mat_Outfit_Primary', (0.92, 0.82, 0.88, 1.0), roughness=0.6)
    obj.data.materials.append(mat_outfit)
    return obj

def create_outfit_casual(name):
    bm = bmesh.new()
    # Casual T-Shirt
    add_cylinder_segment(bm, 0.19, 0.14, 0.64, 0.36, 0.0, 0.0, segs=14)
    for side in [1.0, -1.0]:
        add_capsule_limb(bm, (side * 0.18, 0.0, 0.58), (side * 0.32, 0.0, 0.58), 0.042, segs=8)
    # Denim Jeans
    for side in [1.0, -1.0]:
        add_capsule_limb(bm, (side * 0.12, 0.0, 0.36), (side * 0.14, 0.01, -0.45), 0.072, segs=10)
        add_capsule_limb(bm, (side * 0.14, 0.01, -0.45), (side * 0.14, -0.02, -0.80), 0.052, segs=10)

    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    mat_outfit = create_anime_material('Mat_Outfit_Primary', (0.25, 0.42, 0.68, 1.0), roughness=0.6)
    obj.data.materials.append(mat_outfit)
    return obj

# 6. Shoes Builders
def create_shoes_loafers(name):
    bm = bmesh.new()
    for side in [1.0, -1.0]:
        # High socks
        add_capsule_limb(bm, (side * 0.14, -0.02, -0.55), (side * 0.14, -0.02, -0.84), 0.042, segs=8)
        # Loafer shoe
        add_capsule_limb(bm, (side * 0.14, -0.02, -0.84), (side * 0.14, 0.12, -0.98), 0.046, segs=8)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    mat_shoes = create_anime_material('Mat_Shoes', (0.10, 0.08, 0.07, 1.0), roughness=0.3)
    obj.data.materials.append(mat_shoes)
    return obj

def create_shoes_sneakers(name):
    bm = bmesh.new()
    for side in [1.0, -1.0]:
        # Chunky anime sneaker
        add_capsule_limb(bm, (side * 0.14, -0.02, -0.78), (side * 0.14, -0.02, -0.86), 0.052, segs=8)
        add_capsule_limb(bm, (side * 0.14, -0.02, -0.86), (side * 0.14, 0.13, -0.98), 0.055, segs=8)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    mat_shoes = create_anime_material('Mat_Shoes', (0.95, 0.95, 0.95, 1.0), roughness=0.5)
    obj.data.materials.append(mat_shoes)
    return obj

# 7. Accessories Builders
def create_accessory_glasses(name):
    bm = bmesh.new()
    for side in [1.0, -1.0]:
        # Frame rims
        add_cylinder_segment(bm, 0.032, 0.032, 0.865, 0.825, side * 0.075, 0.155, segs=8)
    # Bridge
    add_capsule_limb(bm, (-0.04, 0.156, 0.845), (0.04, 0.156, 0.845), 0.005, segs=4)
    # Temple arms
    for side in [1.0, -1.0]:
        add_capsule_limb(bm, (side * 0.105, 0.15, 0.845), (side * 0.15, -0.01, 0.85), 0.004, segs=4)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    mat_acc = create_anime_material('Mat_Accessories', (0.10, 0.10, 0.10, 1.0), roughness=0.2)
    obj.data.materials.append(mat_acc)
    return obj

def create_accessory_catears(name):
    bm = bmesh.new()
    # Headband
    for i in range(12):
        ang = (i / 11.0) * math.pi
        x = math.cos(ang) * 0.185
        z = 0.87 + math.sin(ang) * 0.185
        if i < 11:
            ang_next = ((i + 1) / 11.0) * math.pi
            x_next = math.cos(ang_next) * 0.185
            z_next = 0.87 + math.sin(ang_next) * 0.185
            add_capsule_limb(bm, (x, 0.0, z), (x_next, 0.0, z_next), 0.006, segs=4)

    # Fluffy Cat Ears
    for side in [1.0, -1.0]:
        base = mathutils.Vector((side * 0.12, 0.0, 1.02))
        tip = mathutils.Vector((side * 0.16, 0.01, 1.14))
        inner = mathutils.Vector((side * 0.07, 0.0, 1.03))
        v1 = bm.verts.new(base)
        v2 = bm.verts.new(tip)
        v3 = bm.verts.new(inner)
        bm.faces.new([v1, v2, v3])
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    mat_acc = create_anime_material('Mat_Accessories', (0.95, 0.85, 0.90, 1.0), roughness=0.5)
    obj.data.materials.append(mat_acc)
    return obj

# -----------------------------------------------------------------------------
# Actions / Animation Creator
# -----------------------------------------------------------------------------
def add_avatar_actions(arm_obj):
    arm_obj.animation_data_create()

    def insert_rot(pb_name, frame, rx, ry, rz):
        pb = arm_obj.pose.bones.get(pb_name)
        if not pb: return
        pb.rotation_mode = 'XYZ'
        pb.rotation_euler = (rx, ry, rz)
        pb.keyframe_insert(data_path='rotation_euler', frame=frame)

    # 1. Idle (60 frames)
    idle = bpy.data.actions.new(name='Idle')
    arm_obj.animation_data.action = idle
    for f in [1, 60]:
        insert_rot('Chest', f, 0, 0, 0)
        insert_rot('Spine', f, 0, 0, 0)
        insert_rot('Head',  f, 0, 0, 0)
    insert_rot('Chest', 30, 0.06, 0, 0)
    insert_rot('Spine', 30, 0.03, 0, 0)
    insert_rot('Head',  30, -0.04, 0, 0)

    # 2. Walk (40 frames)
    walk = bpy.data.actions.new(name='Walk')
    arm_obj.animation_data.action = walk
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

    # 3. Dance (60 frames)
    dance = bpy.data.actions.new(name='Dance')
    arm_obj.animation_data.action = dance
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

    # 4. Wave (40 frames)
    wave = bpy.data.actions.new(name='Wave')
    arm_obj.animation_data.action = wave
    insert_rot('RightUpperArm', 1, 0.2, 0, -1.8)
    insert_rot('RightUpperArm', 40, 0.2, 0, -1.8)
    for f, r_fore_z in [(1, -0.2), (10, -0.7), (20, -0.2), (30, -0.7), (40, -0.2)]:
        insert_rot('RightLowerArm', f, 0, 0, r_fore_z)

    # 5. Pose (Cute Idol Peace Pose)
    pose = bpy.data.actions.new(name='Pose')
    arm_obj.animation_data.action = pose
    insert_rot('Head', 1, 0.05, -0.15, 0.18)
    insert_rot('Hips', 1, 0.05, 0.12, -0.08)
    insert_rot('RightUpperArm', 1, 0.35, 0.20, -1.45)
    insert_rot('RightLowerArm', 1, 1.20, 0.00, -0.45)
    insert_rot('LeftUpperArm', 1, -0.15, 0.00, 0.35)
    insert_rot('LeftUpperLeg', 1, -0.10, 0.00, 0.05)
    insert_rot('RightUpperLeg', 1, 0.15, 0.00, -0.08)

    arm_obj.animation_data.action = idle

# -----------------------------------------------------------------------------
# Main Assembly & GLB Export Pipeline
# -----------------------------------------------------------------------------
def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)

    print("1. Creating Standard Anime Armature...")
    arm_obj, bones_def = build_armature()

    print("2. Generating Anime Base Bodies (Female & Male)...")
    body_female = create_anime_body('Body_Female', gender='female')
    body_male = create_anime_body('Body_Male', gender='male')

    print("3. Generating Anime Eye Feature Decals...")
    eyes = create_anime_eyes('Face_Eyes')

    print("4. Generating Anime Hairstyles...")
    hair_twintails = create_hair_twintails('Hair_Twintails')
    hair_bob = create_hair_bob('Hair_Bob')
    hair_hime = create_hair_hime('Hair_Hime')
    hair_spiky = create_hair_spiky('Hair_Spiky')
    hair_parted = create_hair_parted('Hair_Parted')

    print("5. Generating Anime Outfits...")
    outfit_sailor = create_outfit_sailor('Outfit_Sailor')
    outfit_blazer = create_outfit_blazer('Outfit_Blazer')
    outfit_hoodie = create_outfit_hoodie('Outfit_Hoodie')
    outfit_casual = create_outfit_casual('Outfit_Casual')

    print("6. Generating Shoes & Accessories...")
    shoes_loafers = create_shoes_loafers('Shoes_Loafers')
    shoes_sneakers = create_shoes_sneakers('Shoes_Sneakers')
    acc_glasses = create_accessory_glasses('Accessory_Glasses')
    acc_catears = create_accessory_catears('Accessory_CatEars')

    print("7. Rigging & Binding All Modular Meshes to Armature...")
    apply_weights_to_mesh(body_female, arm_obj, bones_def)
    apply_weights_to_mesh(body_male, arm_obj, bones_def)
    apply_weights_to_mesh(eyes, arm_obj, bones_def, is_head_only=True)

    apply_weights_to_mesh(hair_twintails, arm_obj, bones_def, is_hair=True)
    apply_weights_to_mesh(hair_bob, arm_obj, bones_def, is_head_only=True)
    apply_weights_to_mesh(hair_hime, arm_obj, bones_def)
    apply_weights_to_mesh(hair_spiky, arm_obj, bones_def, is_head_only=True)
    apply_weights_to_mesh(hair_parted, arm_obj, bones_def, is_head_only=True)

    apply_weights_to_mesh(outfit_sailor, arm_obj, bones_def)
    apply_weights_to_mesh(outfit_blazer, arm_obj, bones_def)
    apply_weights_to_mesh(outfit_hoodie, arm_obj, bones_def)
    apply_weights_to_mesh(outfit_casual, arm_obj, bones_def)

    apply_weights_to_mesh(shoes_loafers, arm_obj, bones_def)
    apply_weights_to_mesh(shoes_sneakers, arm_obj, bones_def)

    apply_weights_to_mesh(acc_glasses, arm_obj, bones_def, is_head_only=True)
    apply_weights_to_mesh(acc_catears, arm_obj, bones_def, is_head_only=True)

    print("8. Adding Embedded Character Actions...")
    add_avatar_actions(arm_obj)

    out_glb = 'z:/home/edwintai/github/3dmodel/anime_avatar_modular.glb'
    print(f"9. Exporting Modular Character Master GLB: {out_glb}...")
    bpy.ops.export_scene.gltf(
        filepath=out_glb,
        export_format='GLB',
        export_skins=True,
        export_animations=True,
        export_animation_mode='ACTIONS'
    )

    size_mb = os.path.getsize(out_glb) / (1024 * 1024)
    print(f"\n✨ SUCCESS! Anime Avatar Modular Master GLB exported: {size_mb:.2f} MB")

if __name__ == '__main__':
    main()
