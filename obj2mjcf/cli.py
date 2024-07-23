"""A CLI for processing composite Wavefront OBJ files for use in MuJoCo."""

import logging
import os
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional
import json 

import trimesh
import tyro
from PIL import Image
from termcolor import cprint

from obj2mjcf import constants
from obj2mjcf.material import Material
from obj2mjcf.mjcf_builder import MJCFBuilder
from obj2mjcf.utils import convert_unity_json_to_nested_dict, split_obj_by_geometry
from obj2mjcf.category_utils import categories_joint_map, find_value_by_key_substring

@dataclass(frozen=True)
class CoacdArgs:
    """Arguments to pass to CoACD.

    Defaults and descriptions are copied from: https://github.com/SarahWeiii/CoACD
    """

    preprocess_resolution: int = 50
    """resolution for manifold preprocess (20~100), default = 50"""
    threshold: float = 0.05
    """concavity threshold for terminating the decomposition (0.01~1), default = 0.05"""
    max_convex_hull: int = -1
    """max # convex hulls in the result, -1 for no maximum limitation"""
    mcts_iterations: int = 100
    """number of search iterations in MCTS (60~2000), default = 100"""
    mcts_max_depth: int = 3
    """max search depth in MCTS (2~7), default = 3"""
    mcts_nodes: int = 20
    """max number of child nodes in MCTS (10~40), default = 20"""
    resolution: int = 2000
    """sampling resolution for Hausdorff distance calculation (1e3~1e4), default = 2000"""
    pca: bool = False
    """enable PCA pre-processing, default = false"""
    seed: int = 0
    """random seed used for sampling, default = 0"""


@dataclass(frozen=True)
class Args:
    obj_dir: str
    """path to a directory containing obj files. All obj files in the directory will be
    converted"""
    texture_dir: str = ""
    """path to a directory containing texture files."""
    obj_filter: Optional[str] = None
    """only convert obj files matching this regex"""
    save_mjcf: bool = False
    """save an example XML (MJCF) file"""
    compile_model: bool = False
    """compile the MJCF file to check for errors"""
    verbose: bool = False
    """print verbose output"""
    decompose: bool = False
    """approximate mesh decomposition using CoACD"""
    coacd_args: CoacdArgs = field(default_factory=CoacdArgs)
    """arguments to pass to CoACD"""
    texture_resize_percent: float = 1.0
    """resize the texture to this percentage of the original size"""
    overwrite: bool = False
    """overwrite previous run output"""
    add_free_joint: bool = False
    """add a free joint to the root body"""
    category: str = ""


def resize_texture(filename: Path, resize_percent) -> None:
    """Resize a texture to a percentage of its original size."""
    if resize_percent == 1.0:
        return
    image = Image.open(filename)
    new_width = int(image.size[0] * resize_percent)
    new_height = int(image.size[1] * resize_percent)
    logging.info(f"Resizing {filename} to {new_width}x{new_height}")
    image = image.resize((new_width, new_height), Image.LANCZOS)
    image.save(filename)


def decompose_convex(filename: Path, work_dir: Path, coacd_args: CoacdArgs) -> bool:
    cprint(f"Decomposing {filename}", "yellow")

    import coacd  # noqa: F401

    obj_file = filename.resolve()
    logging.info(f"Decomposing {obj_file}")

    mesh = trimesh.load(obj_file, force="mesh")
    mesh = coacd.Mesh(mesh.vertices, mesh.faces)  # type: ignore

    parts = coacd.run_coacd(
        mesh=mesh,
        **asdict(coacd_args),
    )

    mesh_parts = []
    for vs, fs in parts:
        mesh_parts.append(trimesh.Trimesh(vs, fs))

    # Save the decomposed parts as separate OBJ files.
    for i, p in enumerate(mesh_parts):
        submesh_name = work_dir / f"{obj_file.stem}_collision_{i}.obj"
        p.export(submesh_name.as_posix())

    return True


def process_obj(filename: Path, args: Args) -> None:
    # Create a directory with the same name as the OBJ file. The processed submeshes
    # and materials will be stored there.
    work_dir = filename.parent / filename.stem
    if work_dir.exists():
        if not args.overwrite:
            proceed = input(
                f"{work_dir.resolve()} already exists, maybe from a previous run? "
                "Proceeding will overwrite it.\nDo you wish to continue [y/n]: "
            )
            if proceed.lower() != "y":
                return
        shutil.rmtree(work_dir)
    work_dir.mkdir(exist_ok=True)
    logging.info(f"Saving processed meshes to {work_dir}")

    # Check if the OBJ files references an MTL file.
    # TODO(kevin): Should we support multiple MTL files?
    process_mtl = False
    with open(filename, "r") as f:
        for line in f.readlines():
            if line.startswith("mtllib"):  # Deals with commented out lines.
                process_mtl = True
                name = line.split()[1]
                break

    sub_mtls: List[List[str]] = []
    mtls: List[Material] = []
    if process_mtl:
        # Make sure the MTL file exists. The MTL filepath is relative to the OBJ's.
        mtl_filename = filename.parent / name
        if not mtl_filename.exists():
            raise RuntimeError(
                f"The MTL file {mtl_filename.resolve()} referenced in the OBJ file "
                f"{filename} does not exist"
            )
        logging.info(f"Found MTL file: {mtl_filename}")

        # Parse the MTL file into separate materials.
        with open(mtl_filename, "r") as f:
            lines = f.readlines()
        # Remove comments.
        lines = [
            line for line in lines if not line.startswith(constants.MTL_COMMENT_CHAR)
        ]
        # Remove empty lines.
        lines = [line for line in lines if line.strip()]
        # Remove trailing whitespace.
        lines = [line.strip() for line in lines]
        # Split at each new material definition.
        for line in lines:
            if line.startswith("newmtl"):
                sub_mtls.append([])
            sub_mtls[-1].append(line)
        for sub_mtl in sub_mtls:
            mtls.append(Material.from_string(sub_mtl))
        
        # Process each material.
        for mtl in mtls:
            logging.info(f"Found material: {mtl.name}")
            if mtl.map_Kd is not None:
                texture_path = Path(mtl.map_Kd)
                texture_name = texture_path.name
                src_filename = filename.parent / texture_path
                
                src_filename = args.texture_dir / texture_path
                if not src_filename.exists():
                    raise RuntimeError(
                        f"The texture file {src_filename} referenced in the MTL file "
                        f"{mtl.name} does not exist"
                    )
                    
                # We want a flat directory structure in work_dir.
                dst_filename = work_dir / texture_name
                shutil.copy(src_filename, dst_filename)
                # MuJoCo only supports PNG textures ¯\_(ツ)_/¯.
                if texture_path.suffix.lower() in [".jpg", ".jpeg"]:
                    image = Image.open(dst_filename)
                    os.remove(dst_filename)
                    dst_filename = (work_dir / texture_path.stem).with_suffix(".png")
                    image.save(dst_filename)
                    texture_name = dst_filename.name
                    mtl.map_Kd = texture_name
                resize_texture(dst_filename, args.texture_resize_percent)
        
        mat_dict = {}
        for mtl in mtls:
            mat_dict[mtl.name] = mtl
        logging.info("Done processing MTL file")
        
    # Check if there is JSON File 
    meshes_hierarchy = None
    json_file = filename.parent / f"{filename.stem}.json"
    if json_file.exists():
        logging.info(f"Found JSON file: {json_file}")
        # Load JSON file
        with open(json_file, "r") as f:
            data = json.load(f)
        meshes_hierarchy = convert_unity_json_to_nested_dict(data)
    logging.info("Done processing JSON file", meshes_hierarchy)
    
    # Split OBJ into Sub Meshes
    submesh_filenames, saved_filenames = split_obj_by_geometry(filename)
    logging.info(f"Splitting OBJ into {len(submesh_filenames)} submeshes")
 
    meshes = {}
    mtls = {}
    for i, saved_filename in enumerate(saved_filenames):    
        logging.info("Processing OBJ file with trimesh")
        sub_filename = Path(submesh_filenames[i])
        saved_filename = Path(saved_filename)
        
        mesh = trimesh.load(
            saved_filename,
            split_object=False,
            group_material=False,
            process=False,
            # Note setting this to False is important. Without it, there are a lot of weird
            # visual artifacts in the texture.        
            maintain_order=True, # False NOTE: why did Yejin set to true?
        )
        
        logging.info(f"Components Material: {mesh}")

        
        if isinstance(mesh, trimesh.base.Trimesh):
            # No submeshes, just save the mesh.
            savename = work_dir / f"{saved_filename.stem}_0.obj"
            logging.info(f"Saving mesh {savename}")
            mesh.export(savename.as_posix(), include_texture=True, header=None)

            # Decompose the mesh into convex pieces if desired.
            decomp_success = False
            if args.decompose:
                decomp_success = decompose_convex(savename, work_dir, args.coacd_args)
                logging.info(f"Decomposition {savename} success: {decomp_success}")

        else:
            # NOTE: We do NOT want to group submeshes
            #logging.info("Grouping and saving submeshes by material")
            logging.info("Saving submeshes")
            for i, geom in enumerate(mesh.geometry.values()):  # type: ignore
                savename = work_dir / f"{saved_filename.stem}_{i}.obj"
                logging.info(f"Saving submesh {savename}")
                geom.export(savename.as_posix(), include_texture=True, header=None)

                # Decompose the mesh into convex pieces if desired.
                decomp_success = False
                if args.decompose:
                    decomp_success = decompose_convex(savename, work_dir, args.coacd_args)
                    logging.info(f"Decomposition {savename} success: {decomp_success}")
        meshes[sub_filename.stem] = mesh        
        #TODO: delete original mesh filename 
   
        _mtls = []
        with open(saved_filename, "r") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith("usemtl"):
                mat_name = line.split()[1]
                logging.info(f"Processing material: {mat_name}")
                logging.info(f"Processed materials: {mat_dict}")
                _mtls.append(mat_dict[mat_name])
   
        mtls[sub_filename.stem] = _mtls
        logging.info(f"Processed materials: {_mtls}")
        
        
    # Delete any MTL files that were created during trimesh processing, if any.
    for file in [
        x
        for x in work_dir.glob("**/*")
        if x.is_file() and "material_0" in x.name and not x.name.endswith(".png")
    ]:
        file.unlink()


    # Build an MJCF.
    joint_map = None
    if args.category in categories_joint_map:
        joint_map = find_value_by_key_substring(categories_joint_map[args.category], filename.stem)
    logging.info(f"Building MJCF for category {joint_map}")
        
    builder = MJCFBuilder(filename, meshes, mtls, meshes_hierarchy, decomp_success=decomp_success)
    builder.build(add_free_joint=args.add_free_joint, add_joints=joint_map)

    # Compile and step the physics to check for any errors.
    if args.compile_model:
        builder.compile_model()

    # Dump.
    if args.save_mjcf:
        builder.save_mjcf()


def main() -> None:
    args = tyro.cli(Args, description=__doc__)

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    # Get all obj files in the directory.
    obj_files = list(Path(args.obj_dir).glob("*.obj"))

    # Filter out the ones that don't match the regex filter.
    if args.obj_filter is not None:
        obj_files = [
            x for x in obj_files if re.search(args.obj_filter, x.name) is not None
        ]

    for obj_file in obj_files:
        cprint(f"Processing {obj_file}", "yellow")
        process_obj(obj_file, args)
