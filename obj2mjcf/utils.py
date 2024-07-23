import os

def convert_unity_json_to_nested_dict(json_data):
    # Initialize the resulting dictionary
    result = {}

    # Iterate over each key-value pair
    for pair in json_data["keyValuePairs"]:
        outer_key = pair["outerKey"]
        inner_key = pair["innerKey"]
        value = pair["value"]
        
        # If the outer key does not exist in the result, add it
        if outer_key not in result:
            result[outer_key] = {}
        
        # Add the inner key and its value to the outer key's dictionary
        result[outer_key][inner_key] = value

    return result 

def split_obj_by_geometry(input_filename):
    submeshes_filenames = []
    saved_filenames = []
    with open(input_filename, 'r') as file:
        lines = file.readlines()

    # Directory to save the split OBJ files
    base_dir = os.path.dirname(input_filename)
    base_name = os.path.splitext(os.path.basename(input_filename))[0]

    geometries = {}
    current_geometry = None
    mtllib_line = None
    current_material = None
    
    vertices = []
    texcoords = []
    normals = []

    for line in lines:
        if line.startswith('mtllib '):
            mtllib_line = line
        elif line.startswith('v '):
            vertices.append(line)
        elif line.startswith('vt '):
            texcoords.append(line)
        elif line.startswith('vn '):
            normals.append(line)
        elif line.startswith('g '):
            current_geometry = line.strip().split(' ')[1]
            if current_geometry not in geometries:
                geometries[current_geometry] = {
                    'lines': [],
                    'faces': {},
                    'vertices': [],
                    'texcoords': [],
                    'normals': []
                }
        elif line.startswith('usemtl '):
            current_material = line.strip()
        if current_geometry:
            if line.startswith('f '):
                if current_material not in geometries[current_geometry]['faces']:
                    geometries[current_geometry]['faces'][current_material] = []
                geometries[current_geometry]['faces'][current_material].append(line)
            else:
                geometries[current_geometry]['lines'].append(line)

    previous_vertex_count = 0
    previous_texcoord_count = 0
    previous_normal_count = 0

    for geometry, data in geometries.items():
        vertex_map = {}
        texcoord_map = {}
        normal_map = {}

        new_vertices = []
        new_texcoords = []
        new_normals = []

        # Process vertices
        for i, line in enumerate(vertices):
            if line in data['lines']:
                new_index = len(new_vertices) + 1
                vertex_map[i + 1] = new_index
                new_vertices.append(line)

        # Process texture coordinates
        for i, line in enumerate(texcoords):
            if line in data['lines']:
                new_index = len(new_texcoords) + 1
                texcoord_map[i + 1] = new_index
                new_texcoords.append(line)

        # Process normals
        for i, line in enumerate(normals):
            if line in data['lines']:
                new_index = len(new_normals) + 1
                normal_map[i + 1] = new_index
                new_normals.append(line)

        new_faces = []

        # Process faces
        for material, faces in data['faces'].items():
            new_faces.append(material + '\n')
            for line in faces:
                parts = line.split()
                new_face = []
                for part in parts[1:]:
                    indices = part.split('/')
                    vertex_index = int(indices[0]) #- previous_vertex_count
                    texcoord_index = int(indices[1])# - previous_texcoord_count if len(indices) > 1 and indices[1] else None
                    normal_index = int(indices[2])# - previous_normal_count if len(indices) > 2 and indices[2] else None

                    new_vertex_index = vertex_map[vertex_index]
                    new_texcoord_index = texcoord_map.get(texcoord_index, '') if texcoord_index else ''
                    new_normal_index = normal_map.get(normal_index, '') if normal_index else ''

                    if new_texcoord_index and new_normal_index:
                        new_face.append(f'{new_vertex_index}/{new_texcoord_index}/{new_normal_index}')
                    elif new_texcoord_index:
                        new_face.append(f'{new_vertex_index}/{new_texcoord_index}')
                    elif new_normal_index:
                        new_face.append(f'{new_vertex_index}//{new_normal_index}')
                    else:
                        new_face.append(f'{new_vertex_index}')
                
                new_faces.append(f"f {' '.join(new_face)}\n")

        new_dir = os.path.join(base_dir, base_name)
        os.makedirs(new_dir, exist_ok=True)
        
        output_filename = os.path.join(new_dir, f"{base_name}_{geometry}.obj")

        with open(output_filename, 'w') as out_file:
            if mtllib_line:
                out_file.write(mtllib_line)
            out_file.writelines(new_vertices)
            out_file.writelines(new_texcoords)
            out_file.writelines(new_normals)
            #out_file.writelines(data['lines'])
            out_file.writelines(new_faces)
        
        print(f"Saved geometry '{geometry}' to {output_filename}")
        saved_filenames.append(output_filename)
        submeshes_filenames.append(f"{geometry}.obj")
        previous_vertex_count += len(new_vertices)
        previous_texcoord_count += len(new_texcoords)
        previous_normal_count += len(new_normals)

    return submeshes_filenames, saved_filenames

if "__name__" == "__main__":
    # Example usage
    import json
    input_filename = '/Users/yejink/Repos/mujoco_simulation/assets/ThorAssets/ManipulaTHOR Objects/Doorways/test/Doorway_Door_10.json'
    with open(input_filename, "r") as f:
        data = json.load(f)
        submeshes_map = convert_unity_json_to_nested_dict(data)
    print(submeshes_map)    

    input_filename = '/Users/yejink/Repos/mujoco_simulation/assets/ThorAssets/ManipulaTHOR Objects/Doorways/test/Doorway_Door_10.obj'
    split_obj_by_geometry(input_filename)