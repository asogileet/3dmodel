import bpy
import bmesh
import mathutils
import math
import os

def disassemble():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    src_glb = 'z:/home/edwintai/github/3dmodel/miku_psp_with_bones_fixed.glb'
    print(f"Loading {src_glb}...")
    bpy.ops.import_scene.gltf(filepath=src_glb)

    # 1. Delete unnecessary objects (like Icosphere)
    for obj in list(bpy.data.objects):
        if 'Icosphere' in obj.name:
            oname = obj.name
            bpy.data.objects.remove(obj, do_unlink=True)
            print(f"Removed {oname}.")
    for mesh in list(bpy.data.meshes):
        if 'Icosphere' in mesh.name:
            bpy.data.meshes.remove(mesh, do_unlink=True)
    bpy.data.orphans_purge()

    # 2. Identify Armature
    arm_obj = None
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE':
            arm_obj = obj
            break
    
    if not arm_obj:
        raise RuntimeError("Armature not found!")
    
    arm_obj.name = "Armature_Miku"
    print(f"Found Armature: {arm_obj.name}")

    # 3. Rename existing mesh objects to modular naming conventions
    rename_map = {
        'Object_101': 'Hair_Miku_Twintails',
        'Object_103': 'Head_Miku',
        'Object_100': 'Face_Mouth_Miku',
        'Object_102': 'Accessory_Miku_Tie',
        'Object_96':  'Accessory_Miku_Headset',
        'Object_97':  'Shoes_Miku_Boots',
    }

    for old_name, new_name in rename_map.items():
        if old_name in bpy.data.objects:
            obj = bpy.data.objects[old_name]
            obj.name = new_name
            obj.data.name = new_name + "_Mesh"
            print(f"Renamed {old_name} -> {new_name}")

    # 4. Merge Left & Right Eyes into Face_Eyes_Miku
    eye_l = bpy.data.objects.get('Object_98')
    eye_r = bpy.data.objects.get('Object_99')
    if eye_l and eye_r:
        bpy.ops.object.select_all(action='DESELECT')
        eye_l.select_set(True)
        eye_r.select_set(True)
        bpy.context.view_layer.objects.active = eye_l
        bpy.ops.object.join()
        eye_l.name = 'Face_Eyes_Miku'
        eye_l.data.name = 'Face_Eyes_Miku_Mesh'
        print("Merged Left and Right eyes into Face_Eyes_Miku")

    # 5. Merge Left & Right Sleeves into Outfit_Miku_Sleeves
    sleeve_l = bpy.data.objects.get('Object_94')
    sleeve_r = bpy.data.objects.get('Object_104')
    if sleeve_l and sleeve_r:
        bpy.ops.object.select_all(action='DESELECT')
        sleeve_l.select_set(True)
        sleeve_r.select_set(True)
        bpy.context.view_layer.objects.active = sleeve_l
        bpy.ops.object.join()
        sleeve_l.name = 'Outfit_Miku_Sleeves'
        sleeve_l.data.name = 'Outfit_Miku_Sleeves_Mesh'
        print("Merged Left and Right sleeves into Outfit_Miku_Sleeves")

    # 6. Separate Object_95 into Outfit_Miku_Top and Outfit_Miku_Skirt
    obj_95 = bpy.data.objects.get('Object_95')
    if obj_95:
        # Duplicate for skirt
        skirt_obj = obj_95.copy()
        skirt_obj.data = obj_95.data.copy()
        skirt_obj.name = 'Outfit_Miku_Skirt'
        skirt_obj.data.name = 'Outfit_Miku_Skirt_Mesh'
        bpy.context.collection.objects.link(skirt_obj)

        top_obj = obj_95
        top_obj.name = 'Outfit_Miku_Top'
        top_obj.data.name = 'Outfit_Miku_Top_Mesh'

        skirt_bones = {'Skirt_Back', 'Skirt_SideLeft', 'Skirt_FrontLeft', 'Skirt_SideRight', 'Skirt_FrontRight'}
        skirt_vg_indices = {top_obj.vertex_groups[b].index for b in skirt_bones if b in top_obj.vertex_groups}

        def is_skirt_face(face, mesh_obj):
            skirt_w = 0.0
            top_w = 0.0
            for v in face.verts:
                for g in mesh_obj.data.vertices[v.index].groups:
                    if g.group in skirt_vg_indices:
                        skirt_w += g.weight
                    else:
                        bname = mesh_obj.vertex_groups[g.group].name
                        if any(k in bname for k in ['Chest', 'Spine', 'Neck', 'Shoulder', 'UpperArm', 'LowerArm']):
                            top_w += g.weight
            return skirt_w > top_w

        # Delete skirt from top
        bm_top = bmesh.new()
        bm_top.from_mesh(top_obj.data)
        faces_to_del_top = [f for f in bm_top.faces if is_skirt_face(f, top_obj)]
        bmesh.ops.delete(bm_top, geom=faces_to_del_top, context='FACES')
        bm_top.to_mesh(top_obj.data)
        top_obj.data.update()
        bm_top.free()

        # Delete top from skirt
        bm_skirt = bmesh.new()
        bm_skirt.from_mesh(skirt_obj.data)
        faces_to_del_skirt = [f for f in bm_skirt.faces if not is_skirt_face(f, skirt_obj)]
        bmesh.ops.delete(bm_skirt, geom=faces_to_del_skirt, context='FACES')
        bm_skirt.to_mesh(skirt_obj.data)
        skirt_obj.data.update()
        bm_skirt.free()

        print(f"Separated Object_95 into Outfit_Miku_Top ({len(top_obj.data.polygons)} faces) and Outfit_Miku_Skirt ({len(skirt_obj.data.polygons)} faces)")

    # 7. Add Animation Actions to Armature_Miku
    add_miku_actions(arm_obj)

    # 8. Export modular GLB
    out_glb = 'z:/home/edwintai/github/3dmodel/miku_modular.glb'
    print(f"Exporting modular Miku to {out_glb}...")
    bpy.ops.export_scene.gltf(
        filepath=out_glb,
        export_format='GLB',
        export_skins=True,
        export_animations=True,
        export_animation_mode='ACTIONS'
    )
    
    size_kb = os.path.getsize(out_glb) / 1024
    print(f"✨ Successfully generated {out_glb} ({size_kb:.1f} KB)!")

def add_miku_actions(arm_obj):
    arm_obj.animation_data_create()

    def insert_rot(pb_name, frame, rx, ry, rz):
        pb = arm_obj.pose.bones.get(pb_name)
        if not pb: return
        pb.rotation_mode = 'XYZ'
        pb.rotation_euler = (rx, ry, rz)
        pb.keyframe_insert(data_path='rotation_euler', frame=frame)

    # 1. Idle (60 frames - breathing chest + twintail sway)
    idle = bpy.data.actions.new(name='Idle')
    arm_obj.animation_data.action = idle
    for f in [1, 60]:
        insert_rot('Chest', f, 0, 0, 0)
        insert_rot('Spine', f, 0, 0, 0)
        insert_rot('Head',  f, 0, 0, 0)
        insert_rot('LeftTwintail_1', f, 0, 0, 0)
        insert_rot('RightTwintail_1', f, 0, 0, 0)
        insert_rot('LeftTwintail_2', f, 0, 0, 0)
        insert_rot('RightTwintail_2', f, 0, 0, 0)
    insert_rot('Chest', 30, 0.06, 0, 0)
    insert_rot('Spine', 30, 0.03, 0, 0)
    insert_rot('Head',  30, -0.04, 0, 0)
    insert_rot('LeftTwintail_1', 30, 0.05, 0, 0.08)
    insert_rot('RightTwintail_1', 30, 0.05, 0, -0.08)
    insert_rot('LeftTwintail_2', 30, 0.08, 0, 0.12)
    insert_rot('RightTwintail_2', 30, 0.08, 0, -0.12)

    # 2. Walk (40 frames)
    walk = bpy.data.actions.new(name='Walk')
    arm_obj.animation_data.action = walk
    for f, (l_leg, r_leg, l_knee, r_knee, l_arm, r_arm, tw_y) in [
        (1,  ( 0.45, -0.45,  0.0,   0.6, -0.35,  0.35,  0.15)),
        (10, ( 0.00,  0.00,  0.2,   0.2,  0.00,  0.00,  0.00)),
        (20, (-0.45,  0.45,  0.6,   0.0,  0.35, -0.35, -0.15)),
        (30, ( 0.00,  0.00,  0.2,   0.2,  0.00,  0.00,  0.00)),
        (40, ( 0.45, -0.45,  0.0,   0.6, -0.35,  0.35,  0.15)),
    ]:
        insert_rot('LeftUpperLeg',  f, l_leg, 0, 0)
        insert_rot('RightUpperLeg', f, r_leg, 0, 0)
        insert_rot('LeftLowerLeg',  f, -l_knee, 0, 0)
        insert_rot('RightLowerLeg', f, -r_knee, 0, 0)
        insert_rot('LeftUpperArm',  f, l_arm, 0, 0)
        insert_rot('RightUpperArm', f, r_arm, 0, 0)
        insert_rot('LeftTwintail_1', f, 0.1, tw_y, 0.1)
        insert_rot('RightTwintail_1', f, 0.1, -tw_y, -0.1)

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
        insert_rot('LeftTwintail_1', f, 0.1, -hip_y * 1.5, 0.2)
        insert_rot('RightTwintail_1', f, 0.1, -hip_y * 1.5, -0.2)

    # 4. Wave (40 frames - natural greeting wave high up beside head)
    wave = bpy.data.actions.new(name='Wave')
    arm_obj.animation_data.action = wave
    insert_rot('RightUpperArm', 1, 0.5, 0.0, -1.2)
    insert_rot('RightUpperArm', 40, 0.5, 0.0, -1.2)
    for f, r_val in [(1, 1.2), (10, 1.6), (20, 1.2), (30, 1.6), (40, 1.2)]:
        insert_rot('RightLowerArm', f, r_val, 0.0, 0.0)
    insert_rot('Head', 1, 0.05, -0.12, 0.10)
    insert_rot('Head', 40, 0.05, -0.12, 0.10)

    # 5. Salute (40 frames - right hand to right eyebrow/temple)
    salute = bpy.data.actions.new(name='Salute')
    arm_obj.animation_data.action = salute
    for f in [1, 40]:
        insert_rot('RightUpperArm', f, 0.0, 0.5, -0.3)
        insert_rot('RightLowerArm', f, 2.3, 0.0, 0.0)
        insert_rot('Chest', f, 0.05, 0.0, 0.0)
        insert_rot('Head',  f, -0.04, 0.0, 0.0)

    # 6. Pose (Cute Idol Peace Pose)
    pose = bpy.data.actions.new(name='Pose')
    arm_obj.animation_data.action = pose
    insert_rot('Head', 1, 0.05, -0.15, 0.18)
    insert_rot('Hips', 1, 0.05, 0.12, -0.08)
    insert_rot('RightUpperArm', 1, 0.35, 0.10, -0.85)
    insert_rot('RightLowerArm', 1, 2.10, 0.00, 0.00)
    insert_rot('LeftUpperArm', 1, -0.15, 0.00, 0.35)
    insert_rot('LeftUpperLeg', 1, -0.10, 0.00, 0.05)
    insert_rot('RightUpperLeg', 1, 0.15, 0.00, -0.08)
    insert_rot('LeftTwintail_1', 1, 0.15, 0.1, 0.25)
    insert_rot('RightTwintail_1', 1, 0.15, -0.1, -0.25)

    # 7. Bow (50 frames - respectful Japanese bow with grounded feet)
    bow = bpy.data.actions.new(name='Bow')
    arm_obj.animation_data.action = bow
    for f, sp_x, ch_x, hd_x in [
        (1,  0.0,  0.0,  0.0),
        (20, 0.35, 0.25, 0.10),
        (30, 0.35, 0.25, 0.10),
        (50, 0.0,  0.0,  0.0)
    ]:
        insert_rot('Spine', f, sp_x, 0, 0)
        insert_rot('Chest', f, ch_x, 0, 0)
        insert_rot('Head',  f, hd_x, 0, 0)
        insert_rot('RightUpperArm', f, -sp_x * 0.4, 0, -0.1)
        insert_rot('LeftUpperArm',  f, -sp_x * 0.4, 0,  0.1)
        insert_rot('LeftTwintail_1', f, sp_x * 0.6, 0, 0)
        insert_rot('RightTwintail_1', f, sp_x * 0.6, 0, 0)

    arm_obj.animation_data.action = idle

if __name__ == '__main__':
    disassemble()
