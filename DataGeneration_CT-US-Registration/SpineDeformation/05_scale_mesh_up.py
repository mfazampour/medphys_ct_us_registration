import os
import sys
import argparse
import glob

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="Scales a mesh to 0.01 of its original size and centers it")

    arg_parser.add_argument(
        "--root_path_vertebrae",
        required=True,
        dest="root_path_vertebrae",
        help="Root path of the vertebrae folders"
    )

    arg_parser.add_argument(
        "--list_file_names",
        required=True,
        dest="txt_file",
        help="Txt file that contains all spines that contain all lumbar vertebrae"
    )

    arg_parser.add_argument(
        "--workspace_scale_mesh",
        required=True,
        dest="workspace_scale_mesh",
        help="ImFusion workspace files that scales a mesh to 0.001 of its original size"
    )

    args = arg_parser.parse_args()

    # iterate over spine IDS
    with open(args.txt_file) as file:
        spine_ids = [line.strip() for line in file]

    placeholders = ['PathToFile', 'PathToSave']

    for spine_id in spine_ids:

        # get the paths for the segmentations of all vertebrae belonging to this spine
        unique_identifier = "*/**" + str(spine_id) + "*scaled_deformed*.obj"
        vert_mesh_paths = sorted(glob.glob(os.path.join(args.root_path_vertebrae, unique_identifier), recursive=True))

        for vert_mesh_path in vert_mesh_paths:
            arguments_imfusion = ""
            for p in placeholders:

                if p == 'PathToFile':
                    value = vert_mesh_path

                if p == 'PathToSave':
                    value = vert_mesh_path.replace('scaled_deformed','_deformed')

                arguments_imfusion += p + "=" + value + " "

            print('ARGUMENTS: ', arguments_imfusion)
            os.system("ImFusionConsole" + " " + args.workspace_scale_mesh + " " + arguments_imfusion)
            print('################################################### ')