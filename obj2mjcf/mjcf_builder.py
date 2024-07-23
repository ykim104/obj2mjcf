import logging
from pathlib import Path
from typing import Any, List, Union
from scipy.spatial.transform import Rotation as R
from collections import deque

import mujoco
import numpy as np
import trimesh
from lxml import etree
from termcolor import cprint

from obj2mjcf import constants
from obj2mjcf.material import Material
from obj2mjcf.category_utils import dictionary_to_ordered_nested_tuples

class MJCFBuilder:
    """Builds a MuJoCo XML model from a mesh and materials."""

    def __init__(
        self,
        filename: Path,
        meshes: Union[trimesh.base.Trimesh, Any],
        materials: List[Material],
        meshes_hierarchy: dict = None,
        work_dir: Path = Path(),
        decomp_success: bool = False,
    ):
        self.filename = filename
        self.meshes = meshes
        self.materials = materials
        self.meshes_hierarchy = meshes_hierarchy
        self.decomp_success = decomp_success

        self.work_dir = work_dir
        if self.work_dir == Path():
            self.work_dir = filename.parent / filename.stem

        self.tree = None

    def add_visual_and_collision_default_classes(
        self,
        root: etree.Element,
    ):
        # Define the default element.
        default_elem = etree.SubElement(root, "default")

        # Define visual defaults.
        visual_default_elem = etree.SubElement(default_elem, "default")
        visual_default_elem.attrib["class"] = "visual"
        etree.SubElement(
            visual_default_elem,
            "geom",
            group="2",
            type="mesh",
            contype="0",
            conaffinity="0",
        )

        # Define collision defaults.
        collision_default_elem = etree.SubElement(default_elem, "default")
        collision_default_elem.attrib["class"] = "collision"
        etree.SubElement(collision_default_elem, "geom", group="3", type="mesh")

    def add_assets(self, root: etree.Element, mtls: List[Material]) -> etree.Element:
        # Define the assets element.
        asset_elem = etree.SubElement(root, "asset")

        texture_names  = []
        material_names = []
        # TODO: fix repeating texture names
        for material in mtls:
            logging.info(f"Adding material {material}")
            if material.map_Kd is not None:
                texture = Path(material.map_Kd)
                if texture.stem not in texture_names:
                    # Create the texture asset.
                    etree.SubElement(
                        asset_elem,
                        "texture",
                        type="2d",
                        name=texture.stem,
                        file=texture.name,
                    )
                    texture_names.append(texture.stem)
                    
                # Reference the texture asset in a material asset.
                if material.name not in material_names:
                    etree.SubElement(
                        asset_elem,
                        "material",
                        name=material.name,
                        texture=texture.stem,
                        specular=material.mjcf_specular(),
                        shininess=material.mjcf_shininess(),
                        rgba=material.mjcf_rgba(),
                    )
                    material_names.append(material.name)
            else:
                if material.name not in material_names:
                    etree.SubElement(
                        asset_elem,
                        "material",
                        name=material.name,
                        specular=material.mjcf_specular(),
                        shininess=material.mjcf_shininess(),
                        rgba=material.mjcf_rgba(),
                    )
                    material_names.append(material.name)
        return asset_elem

    def add_visual_geometries(
        self,
        obj_body: etree.Element,
        asset_elem: etree.Element,
        geom, 
        geom_name,
        material_name,
    ):
        # Constants
        filename = self.filename
        mesh = geom
        meshname = Path(geom_name)
        material_name = material_name
        
        materials = self.materials

        process_mtl = len(materials) > 0

        # Add visual geometries to object body.
        if isinstance(mesh, trimesh.base.Trimesh):
            # Add the mesh to assets.
            etree.SubElement(asset_elem, "mesh", file=meshname.as_posix())
            # Add the geom to the worldbody.
            if process_mtl:
                e_ = etree.SubElement(
                    obj_body,
                    "geom",
                    material=material_name,
                    mesh=meshname.stem,
                )
                e_.attrib["class"] = "visual"
            else:
                e_ = etree.SubElement(obj_body, "geom", mesh=meshname.stem)
                e_.attrib["class"] = "visual"
        
    def add_collision_geometries(
        self,
        obj_body: etree.Element,
        asset_elem: etree.Element,
        geom,
        geom_name,
    ):
        # Constants.
        filename = self.filename
        decomp_success = self.decomp_success
        #meshname = Path(f"{filename.stem}.obj")
        mesh = geom #self.mesh
        meshname = Path(geom_name)
        
        work_dir = self.work_dir
        
                    
        
        if decomp_success:
            # Find collision files from the decomposed convex hulls.
            collisions = [
                x
                for x in work_dir.glob("**/*")
                if x.is_file() and "collision" in x.name and meshname.stem in x.name
            ]
            collisions.sort(key=lambda x: int(x.stem.split("_")[-1]))

            for collision in collisions:
                etree.SubElement(asset_elem, "mesh", file=collision.name)
                rgb = np.random.rand(3)  # Generate random color for collision meshes.
                e_ = etree.SubElement(
                    obj_body,
                    "geom",
                    mesh=collision.stem,
                    rgba=f"{rgb[0]} {rgb[1]} {rgb[2]} 1",
                )
                e_.attrib["class"] = "collision"
        else:
            # If no decomposed convex hulls were created, use the original mesh as the
            # collision mesh.
            if isinstance(mesh, trimesh.base.Trimesh):
                meshname = Path(f"{filename.stem}.obj")
                e_ = etree.SubElement(obj_body, "geom", mesh=meshname.stem)
                e_.attrib["class"] = "collision"
            else:
                for i, (name, geom) in enumerate(mesh.geometry.items()):
                    meshname = Path(f"{filename.stem}_{i}.obj")
                    e_ = etree.SubElement(obj_body, "geom", mesh=meshname.stem)
                    e_.attrib["class"] = "collision"

    def build(
        self,
        add_free_joint: bool = False,
        add_joints: dict = None,
    ) -> None:
        # Constants.
        filename = self.filename
        mtls = self.materials

        # Start assembling xml tree.
        root = etree.Element("mujoco", model=filename.stem)
        etree.SubElement(root, "compiler", angle="radian")

        # Add defaults.
        self.add_visual_and_collision_default_classes(root)

        # Add assets.
        all_mats = [m for mtl in mtls.values() for m in mtl]
        asset_elem = self.add_assets(root, all_mats)
        
        # Add worldbody.
        worldbody_elem = etree.SubElement(root, "worldbody")
        obj_body = etree.SubElement(worldbody_elem, "body", name=filename.stem)
        if add_free_joint:
            etree.SubElement(obj_body, "freejoint")

        
        # For adding joints        
        contact_elem = etree.SubElement(root, "contact")
    
        bbox_center = None
        if add_joints is not None:
            # Add contact for adding `exclude` tags
            
            tree = add_joints["tree"]
            self.meshes = dictionary_to_ordered_nested_tuples(self.meshes, tree)
        
            if "bbox_center" in self.meshes_hierarchy.keys():
                bbox_center = self.meshes_hierarchy["bbox_center"]["position"]
                bbox_center =  [float(num) for num in bbox_center.strip("()").split(", ")]
        else:
            self.meshes = list(self.meshes.items())


        q = deque()
        q.append((self.meshes[0], obj_body, list(self.meshes[1:]))) # node, parent, children 

        def add_to_q(item, parent, children):
            if isinstance(item, list):
                add_to_q(item[0], parent, item[1:])
            else:
                q.append((item, parent, children))
            
        i=0    
        while q: 
            logging.info(i)
            node = q.popleft()
            current, obj_body, children = node # q.popleft()
            if current is None:
                continue 
            
            if isinstance(current, list):
                for c in current:
                    add_to_q(c, obj_body, children)
            else:             
                # leaf node 
                (name, mesh) = current
                
                str_pos = "0 0 0"
                str_quat = "1 0 0 0"
                if "mirror" in add_joints.keys() and name in add_joints["mirror"].keys():
                    #NOTE: Specific to Doorway_Double assets
                    mname = add_joints["mirror"][name]
                    localParentPosition = self.meshes_hierarchy[mname]["localParentPosition"]
                    localParentRotation = self.meshes_hierarchy[mname]["localParentRotation"]
                    manme_pos =  [float(num) for num in localParentPosition.strip("()").split(", ")]

                    pos = np.array(manme_pos) #np.array(bbox_center)*2 
                    pos[0] = -1*(bbox_center[0]*2 - manme_pos[0])
                    
                    # stringify
                    
                    str_pos = ' '.join(map(str, pos))
                    str_quat = "0.0007963 0 0.9999997 0" # 180 degrees rotation around y-axis
                else:
                    if self.meshes_hierarchy is not None:
                        logging.info(f"name : {self.meshes_hierarchy[name].keys()}")
                        localParentPosition = self.meshes_hierarchy[name]["localParentPosition"]
                        localParentRotation = self.meshes_hierarchy[name]["localParentRotation"]

                        #unity to mujoco
                        pos =  [float(num) for num in localParentPosition.strip("()").split(", ")]
                        if i==0 and bbox_center is not None: # only affects root
                            pos = np.array(pos) - np.array(bbox_center)
                        pos[0] *= -1                    
                        
                        #unity to mujoco
                        euler_deg =  [float(num) for num in localParentRotation.strip("()").split(", ")]
                        quat = R.from_euler('xyz', euler_deg, degrees=True).as_quat()
                        quat = quat[[3, 0, 1, 2]]
                        
                        # stringify
                        str_pos = ' '.join(map(str, pos))
                        str_quat = ' '.join(map(str, quat))
                        
    
                sub_obj_body = etree.SubElement(obj_body, "body", name=f"{filename.stem}_{name}", pos=str_pos, quat=str_quat)
            
                if add_joints is not None and name in add_joints.keys():
                    logging.info(f"Adding joint for {name}")
                    joint_map = add_joints[name]
                    etree.SubElement(sub_obj_body, "joint", type=joint_map["type"], axis=joint_map["axis"], range=joint_map["range"])
                    etree.SubElement(contact_elem, "exclude", body1=obj_body.get('name'), body2=f"{filename.stem}_{name}")
                    
                if isinstance(mesh, trimesh.base.Trimesh):
                    meshname = f"{filename.stem}_{name}_{0}.obj"
                    logging.info(f"Adding mesh {meshname}")
        
                    # Add visual and collision geometries to object body.
                    self.add_visual_geometries(sub_obj_body, asset_elem, mesh, meshname, mtls[name][0].name)
                    self.add_collision_geometries(sub_obj_body, asset_elem, mesh, meshname)

                else:
                    logging.info(f"Adding mesh {mesh}")
                    for i, (_, geom) in enumerate(mesh.geometry.items()):
                        meshname = f"{filename.stem}_{name}_{i}.obj"
                        logging.info(f"Adding mesh {meshname}")
                                
                        # Add visual and collision geometries to object body.
                        self.add_visual_geometries(sub_obj_body, asset_elem, geom, meshname, mtls[name][i].name)
                        self.add_collision_geometries(sub_obj_body, asset_elem, geom, meshname)
                obj_body = sub_obj_body
                if isinstance(children, list) and len(children) > 0:
                    logging.info("list child")
                    q.append((children[0], obj_body, children[1:]))    
                else:
                    logging.info("single child")
                    q.append((children, obj_body, None))
            i+=1 
                   
        # Create the tree.
        tree = etree.ElementTree(root)
        etree.indent(tree, space=constants.XML_INDENTATION, level=0)
        self.tree = tree

    def compile_model(self):
        # Constants.
        filename = self.filename
        work_dir = self.work_dir

        # Pull up tree if possible.
        tree = self.tree
        if tree is None:
            raise ValueError("Tree has not been defined yet.")

        # Create the work directory if it does not exist.
        try:
            tmp_path = work_dir / "tmp.xml"
            tree.write(tmp_path, encoding="utf-8")
            model = mujoco.MjModel.from_xml_path(tmp_path.as_posix())
            data = mujoco.MjData(model)
            mujoco.mj_step(model, data)
            cprint(f"{filename} compiled successfully!", "green")
        except Exception as e:
            cprint(f"Error compiling model: {e}", "red")
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def save_mjcf(
        self,
    ):
        # Constants.
        filename = self.filename
        work_dir = self.work_dir

        # Pull up tree if possible.
        tree = self.tree
        if tree is None:
            raise ValueError("Tree has not been defined yet.")

        # Save the MJCF file.
        xml_path = work_dir / f"{filename.stem}.xml"
        logging.info(f"Saving MJCF to {xml_path}")
        tree.write(xml_path.as_posix(), encoding="utf-8")
        logging.info(f"Saved MJCF to {xml_path}")
