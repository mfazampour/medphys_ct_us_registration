import os
import sys
import argparse
import glob
import open3d as o3d
from pathlib import Path


def get_name_deformed_spine(spine_id, deform):
    return spine_id + "forcefield" + str(deform) + "_lumbar_deformed.obj"

def get_name_deformed_vertebra(spine_id,verLev,deform):
    return spine_id + "_verLev" + str(verLev) + "_forces" + str(deform) + "_deformed_20_0.obj"

def get_vertebrae_meshes_deformed_paths(vertebrae_dir, spine_id,deform):
    vertebrae_meshes = []

    # get all 5 vertebrae paths for one spine id and one deformation
    for i in range(20,25):
        curr_vert_path = os.path.join(vertebrae_dir,spine_id + "_verLev" + str(i),get_name_deformed_vertebra(spine_id, i, deform ))
        if(not Path(curr_vert_path).is_file()):
            print("No deformed vertebra found for spine %s, vert %s and deform %d" % (spine_id, str(i), deform), file=sys.stderr)
        vertebrae_meshes.append(curr_vert_path)
    return vertebrae_meshes

def center(spine_path, vert_paths):
    # read mesh
    spine = o3d.io.read_triangle_mesh(spine_path)

    # get center
    centerSpine = spine.get_center()

    # move to center
    vertsSpine = spine.vertices - centerSpine
    spine.vertices = o3d.utility.Vector3dVector(vertsSpine)

    # save mesh
    o3d.io.write_triangle_mesh(spine_path.replace("_lumbar_deformed.obj", "_lumbar_deformed_centered.obj"), spine)

    # move also all of the vertebrae meshes to the center
    for vert_path in vert_paths:
        curr_vert = o3d.io.read_triangle_mesh(vert_path)
        vertices_curr_vert = curr_vert.vertices - centerSpine
        curr_vert.vertices = o3d.utility.Vector3dVector(vertices_curr_vert)
        o3d.io.write_triangle_mesh(vert_path.replace("_deformed_", "_deformed_centered_"), curr_vert)

def center_all_deformed_spines_and_vertebrae_in_a_list(txt_file,nr_deform_per_spine,root_path_spines,root_path_vertebrae):
    # iterate over spine IDS
    with open(txt_file) as file:
        spine_ids = [line.strip() for line in file]

    for spine_id in spine_ids:
        for deform in range(int(nr_deform_per_spine)):
            print("Centering the spine and vertebrae of: " + str(spine_id) + " and deform " + str(deform))
            # get the paths to the spine with current spine_id and deformation number
            spine_mesh_path = os.path.join(root_path_spines, spine_id, get_name_deformed_spine(spine_id, deform))
            if (not Path(spine_mesh_path).is_file()):
                print("No deformed mesh found for spine %s and deform %d" % (spine_id, deform), file=sys.stderr)

            vertebrae_mesh_paths = get_vertebrae_meshes_deformed_paths(root_path_vertebrae, spine_id, deform)

            center(spine_mesh_path, vertebrae_mesh_paths)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="Center mesh")

    arg_parser.add_argument(
        "--root_path_spines",
        required=True,
        dest="root_path_spines",
        help="Root path of the spine folders"
    )

    arg_parser.add_argument(
        "--root_path_vertebrae",
        required=True,
        dest="root_path_vertebrae",
        help="Root path of the spine folders"
    )

    arg_parser.add_argument(
        "--list_file_names",
        required=True,
        dest="txt_file",
        help="Txt file that contains all spines that contain all lumbar vertebrae"
    )

    arg_parser.add_argument(
        "--nr_deform_per_spine",
        required=True,
        dest="nr_deform_per_spine",
        help="Number of deformed spines per initial spine."
    )

    args = arg_parser.parse_args()

    center_all_deformed_spines_and_vertebrae_in_a_list(txt_file=args.txt_file,nr_deform_per_spine=args.nr_deform_per_spine,root_path_vertebrae=args.root_path_vertebrae, root_path_spines=args.root_path_spines)

    """
    spine_path = "/home/miruna20/Documents/Thesis/Dataset/Patients/CT_segm/spines/patient5_ct/patient5_mesh.obj"
    vert_paths = ["/home/miruna20/Documents/Thesis/Dataset/Patients/CT_segm/vertebrae/patient5_ct_verLev20/patient5_ct_seg_verLev20_msh.obj",
                  "/home/miruna20/Documents/Thesis/Dataset/Patients/CT_segm/vertebrae/patient5_ct_verLev21/patient5_ct_seg_verLev21_msh.obj",
                  "/home/miruna20/Documents/Thesis/Dataset/Patients/CT_segm/vertebrae/patient5_ct_verLev22/patient5_ct_seg_verLev22_msh.obj",
                  "/home/miruna20/Documents/Thesis/Dataset/Patients/CT_segm/vertebrae/patient5_ct_verLev23/patient5_ct_seg_verLev23_msh.obj",
                  "/home/miruna20/Documents/Thesis/Dataset/Patients/CT_segm/vertebrae/patient5_ct_verLev24/patient5_ct_seg_verLev24_msh.obj"
                  ]
    center(spine_path,vert_paths)
    """



