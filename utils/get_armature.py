import bpy
import json


def get_armature_data(armature_name, json_file_path='data/armature_configuration/relevant_armature.json'):
    armature = bpy.data.objects[armature_name]
    if armature is None:
        raise Exception(armature)

    with open(json_file_path, 'r') as infile:
        relevant_bones = json.load(infile)

    print("i am here")

    bones_data = {}

    for bone_name in relevant_bones["relevant_bones"]:
        bone = armature.pose.bones.get(bone_name)
        if bone:
            # Store data: location, rotation
            bones_data[bone_name] = {
                'location': list(bone.head),  # Convert Vector to list
                'rotation': list(bone.rotation_quaternion)  # Convert Quaternion to list
            }

    # In your get_armature.py script
    print('---JSON_START---')
    print(json.dumps(bones_data))  # Your JSON data
    print('---JSON_END---')

    if bones_data is None:
        raise Exception("No bones found in armature data.")


get_armature_data('rig')
