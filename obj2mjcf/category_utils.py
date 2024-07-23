
DOOR_JOINT = {
                "type": "hinge",
                "axis": "0 1 0",
                "range": "-1.57 1.57",
            }
DOOR_MIRROR_JOINT = DOOR_JOINT.copy()
DOOR_MIRROR_JOINT["axis"] = "0 -1 0"
#DOUBLE_DOOR_JOINT["axis"] = "0 -1 0"

DOORHANDLE_JOINT = {}

categories_joint_map = {
    "doorway":  {
        "Doorway" : {
            "tree": ["mesh_2", "mesh_1", "mesh_0"],
            "mesh_1": DOOR_JOINT}, 
            #"tree": ["mesh_2", "mesh_0", "mesh_1"],
            #"mesh_0": DOOR_JOINT}, 
        "Doorway_Door": {
            #"tree": ["mesh_0", "mesh_1"],
            #"mesh_0": DOOR_JOINT},
            "tree": ["mesh_1", "mesh_0"],
            "mesh_1": DOOR_JOINT},
        "Doorway_Double": {
            #"tree": ["mesh_4", [["mesh_0", "mesh_1"], ["mesh_2", "mesh_3"]]],
            #"mesh_0": DOOR_JOINT, 
            #"mesh_2": DOOR_MIRROR_JOINT,
            #"mirror": {"mesh_2":"mesh_0"}},
            "tree": ["mesh_4", [["mesh_1", "mesh_0"], ["mesh_3", "mesh_2"]]],
            "mesh_1": DOOR_JOINT, 
            "mesh_3": DOOR_MIRROR_JOINT,
            "mirror": {"mesh_3":"mesh_1"}},
    }
}

def find_value_by_key_substring(data, search_key):
    l = 0 
    final_value = None 
    for key, value in data.items():
        print(f"key: {key}, value: {value}")
        if search_key.startswith(key):
            if len(key) > l:
                final_value = value
                l = len(key)
            
    return final_value 

# Function to convert dictionary to nested tuples
def dictionary_to_ordered_nested_tuples(dictionary, tree):
    def recursive_convert(subtree):
        if isinstance(subtree, list):
            return list(recursive_convert(item) for item in subtree)
        else:
            return (subtree, dictionary.get(subtree))

    return recursive_convert(tree)