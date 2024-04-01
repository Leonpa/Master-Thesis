import bpy
import json
import random


def apply_perturbations_within_limits(armature_name, base_data_path='data/armature_configuration/idle_armature.json',
                                      constraints_path='data/armature_configuration/armature_boundaries.json'):
    # Load base bone data from the JSON file
    with open(base_data_path, 'r') as infile:
        base_data = json.load(infile)

    # Load bone constraints from the JSON file
    with open(constraints_path, 'r') as infile:
        constraints = json.load(infile)

    # Get the armature object
    armature = bpy.data.objects[armature_name]
    bpy.context.view_layer.objects.active = armature  # Set the armature as active
    bpy.ops.object.mode_set(mode='POSE')  # Switch to Pose Mode

    # Iterate through the bones and apply perturbations within limits
    for bone_name, attrs in base_data.items():
        bone = armature.pose.bones.get(bone_name)
        if bone:
            # Set the bone's base position and rotation
            bone.location = attrs['location']
            bone.rotation_quaternion = attrs['rotation']

            # Check if there are constraints for this bone
            if bone_name in constraints:
                # Apply perturbations within the specified limits
                for i in range(3):  # For x, y, and z in location
                    max_trans = constraints[bone_name]['max_translation'][i]
                    bone.location[i] = (random.random() - 0.5) * 2 * max_trans

                for i in range(4):  # For quaternion rotation (x, y, z, w)
                    max_rot = constraints[bone_name]['max_rotation'][i]
                    bone.rotation_quaternion[i] += (random.random() - 0.5) * 2 * max_rot
            else:
                print(f"No constraints found for bone {bone_name}. Using base position and rotation.")
        else:
            print(f"Bone {bone_name} not found in armature {armature_name}.")

    # Return to Object Mode
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.wm.save_mainfile()


# Usage example:
apply_perturbations_within_limits('rig')
