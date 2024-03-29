import bpy
import os


def create_fbx(motion_vector=None, fbx_filepath='fbx_profiles/mousey.fbx', character_name='mousey.fb', armature_name='rig',
               animation_start_frame=1, animation_end_frame=50):
    """
    This function creates an FBX file with the specified motion vector

    :param motion_vector:
    :param fbx_filepath:
    :param character_name:
    :param armature_name:
    :param animation_start_frame:
    :param animation_end_frame:
    :return:
    """
    # Filepaths
    dir_path = os.path.dirname(os.path.realpath(__file__))
    fbx_filepath = os.path.join(dir_path, fbx_filepath)
    output_filepath = os.path.join(dir_path, "output/")

    # Clear existing objects and load the FBX file
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=fbx_filepath)
    armature = bpy.data.objects.get(armature_name)

    if armature:
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode='POSE')
    else:
        print(f"No armature named {armature_name} found.")
        exit()

    # Define the motion vector
    motion_vector = {
        'lip.T.L.001': (0, 0, 0.8),     # Adjust bone names and values as necessary
        'lip.T.R.001': (0, 0, -0.8),
        'lips.L': (0, 0, 0.8),
        'lips.R': (0, 0, -0.8),
    }

    # Apply initial pose
    for bone_name, rotation in motion_vector.items():
        bone = armature.pose.bones.get(bone_name)
        if bone:
            bone.rotation_mode = 'XYZ'          # Ensure rotation mode is consistent
            bpy.context.scene.frame_set(animation_start_frame)
            bone.rotation_euler = (0, 0, 0)     # Initial neutral position
            bone.keyframe_insert(data_path="rotation_euler", frame=animation_start_frame)
        else:
            print(f"No bone named {bone_name} found in the armature.")

    # Animate the motion that is defined in the motion vector
    for frame in range(animation_start_frame + 1, animation_end_frame + 1):
        bpy.context.scene.frame_set(frame)
        for bone_name, target_rotation in motion_vector.items():
            bone = armature.pose.bones.get(bone_name)
            if bone:
                fraction = (frame - animation_start_frame) / (animation_end_frame - animation_start_frame)   # Interpolate rotation for current frame
                interpolated_rotation = [0, 0, fraction * target_rotation[2]]  # Modify Z rotation based on fraction
                bone.rotation_euler = interpolated_rotation
                bone.keyframe_insert(data_path="rotation_euler", frame=frame)

    # Return to object mode and export as FBX
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.export_scene.fbx(filepath=(output_filepath + character_name), use_selection=False)


if __name__ == "__main__":
    create_fbx()
