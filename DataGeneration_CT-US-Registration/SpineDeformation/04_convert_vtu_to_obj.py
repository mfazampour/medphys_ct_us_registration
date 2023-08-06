import meshio
import numpy as np
import os
import argparse
import glob
import sys

def vtuToObj(file_path):
    """
    Transform vtu file resulted from sofa framework to obj file
    :param file_path:
    :return:
    """
    mesh_vtu = meshio.read(file_path)

    mesh = meshio.Mesh(
       # mesh_vtu.points * 1e3,
        mesh_vtu.points,
        mesh_vtu.cells,
        # Optionally provide extra data on points, cells, etc.
        mesh_vtu.point_data,
        # Each item in cell data must match the cells array
        mesh_vtu.cell_data,
    )

    dst_filename = os.path.join(file_path.replace(".vtu", ".obj"))
    mesh.write(dst_filename)

def vtuToObj_all_files(txt_file, root_path_vertebrae):
    with open(txt_file) as file:
        spine_ids = [line.strip() for line in file]

    for spine_id in spine_ids:
        # gather all vtu files belonging to this spine
        print("Processing: " + str(spine_id))
        look_for = "**/*" + str(spine_id) +  '*.vtu'
        filenames = glob.glob(os.path.join(root_path_vertebrae, look_for),recursive=True)
        if(len(filenames)%5!=0):
            print("The number of vtu files to be found needs to be a multiple of 5: " + str(spine_id), file=sys.stderr)
            continue

        for vtu_file in filenames:
            try:
                vtuToObj(vtu_file)
            except Exception:
                print("There was something wrong with transforming from vtu to obj: " + str(vtu_file), file=sys.stderr)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Generate strings in between vertebrae for spine deformation")

    arg_parser.add_argument(
        "--list_file_names",
        required=True,
        dest="txt_file",
        help="Txt file that contains all spines that contain all lumbar vertebrae"
    )

    arg_parser.add_argument(
        "--root_path_vertebrae",
        required=True,
        dest="root_path_vertebrae",
        help="Root path to the vertebrae folders."
    )
    args = arg_parser.parse_args()

    print("Converting all vtu files resulted from spine deformation to obj")
    vtuToObj_all_files(args.txt_file,args.root_path_vertebrae)
    # sanity check
    #vtuToObj("/home/miruna20/Documents/Thesis/SpineDeformation/code/sofa/build/results/spinesub-verse500_vert3_20_0.vtu")
