import argparse
import os
import glob
import sys

if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(description="Extract point clouds from raycasted ultrasound labelmaps")

    arg_parser.add_argument(
        "--list_file_names",
        required=True,
        dest="txt_file",
        help="Txt file that contains all spines that contain all lumbar vertebrae"
    )
    arg_parser.add_argument(
        "--workspace_file",
        required=True,
        dest="workspace_file",
        help="ImFusion workspace files that has all of the necessary algo info to transform the US labelmap into a pointcloud"
    )
    arg_parser.add_argument(
        "--root_path_spines",
        required=True,
        dest="root_path_spines",
        help="Root path to the spine folders."
    )
    arg_parser.add_argument(
        "--nr_deform_per_spine",
        required=True,
        dest="nr_deform_per_spine",
        help="Number of deformations per spine"
    )

    args = arg_parser.parse_args()
    print("Extract point clouds from raycasted ultrasound labelmaps")

    # iterate over the txt file
    with open(args.txt_file) as file:
        spine_ids = [line.strip() for line in file]

    placeholders = ['PathToRaycastedLabels', 'PathToTrackingStream', 'PathToSavePcd']

    for spine_id in spine_ids:
        for deform in range(int(args.nr_deform_per_spine)):
            print("Extracting pointcloud from ultrasound for: " + str(spine_id) + "and deformation" + str(deform))

            # create arguments list to call ImFusion with
            arguments = ""
            dir_name = os.path.join(args.root_path_spines,spine_id)
            for p in placeholders:
                if p == 'PathToRaycastedLabels':
                    value = os.path.join(dir_name,"labels_force" + str(deform),"raycasted")
                elif p == 'PathToTrackingStream':
                    value = os.path.join(dir_name,"tracking_force" + str(deform) + ".ts")
                elif p == 'PathToSavePcd':
                    value = os.path.join(dir_name, spine_id + "_partial_spine_pcd_force" + str(deform) + ".pcd")

                arguments += p + "=" + str(value) + " "

            print('ARGUMENTS: ', arguments)
            os.system("ImFusionConsole" + " " + args.workspace_file + " " + arguments)
            print('################################################### ')