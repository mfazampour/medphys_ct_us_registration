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
        "--root_path_vertebrae",
        required=True,
        dest="root_path_vertebrae",
        help="Root path to the vertebrae folders."
    )
    arg_parser.add_argument(
        "--nr_deform_per_spine",
        required=True,
        dest="nr_deform_per_spine",
        help="Number of deformations per spine"
    )

    args = arg_parser.parse_args()
    print("Extract vertebrae point clouds from raycasted ultrasound labelmaps")

    # iterate over the txt file
    with open(args.txt_file) as file:
        spine_ids = [line.strip() for line in file]

    placeholders = ['PathToRaycastedLabels', 'PathToTrackingStream', 'PathToSavePcd', 'Value1', 'Value2', 'Value3', 'Value4']

    for spine_id in spine_ids:
        for deform in range(int(args.nr_deform_per_spine)):

            # we extract one point cloud per vertebra
            for vertebra in range(20,25):
                values_to_remove = ['13', '14', '15', '16', '17']
                print("Extracting pointcloud from ultrasound for: " + str(spine_id) + "and deformation" + str(deform) + " and vert: " + str(vertebra))

                # the current level of the vertebra we keep, the rest of the labels we replace with 0
                values_to_remove.remove(str(vertebra-7))

                # create arguments list to call ImFusion with
                arguments = ""
                dir_name_spine = os.path.join(args.root_path_spines, spine_id)
                dir_name_vert = os.path.join(args.root_path_vertebrae, spine_id + "_verLev" + str(vertebra))
                for p in placeholders:
                    if p == 'PathToRaycastedLabels':
                        value = os.path.join(dir_name_spine, "labels_force" + str(deform), "raycasted")
                    elif p == 'PathToTrackingStream':
                        value = os.path.join(dir_name_spine, "tracking_force" + str(deform) + ".ts")
                    elif p == 'PathToSavePcd':
                        value = os.path.join(dir_name_vert, spine_id + "_verLev" + str(vertebra) +  "_forces" + str(deform) + "_deformed_clean.pcd")
                    elif p == 'Value1':
                        value = values_to_remove[0]
                    elif p == 'Value2':
                        value = values_to_remove[1]
                    elif p == 'Value3':
                        value = values_to_remove[2]
                    elif p == 'Value4':
                        value = values_to_remove[3]
                    arguments += p + "=" + str(value) + " "

                print('ARGUMENTS: ', arguments)
                os.system("ImFusionConsole" + " " + args.workspace_file + " " + arguments)
                print('################################################### ')