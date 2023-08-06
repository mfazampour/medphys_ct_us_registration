import argparse
import os
import glob
import sys


def process(args):

    # iterate over the txt file
    with open(args.txt_file) as file:
        spine_ids = [line.strip() for line in file]
    placeholders = ['PathToRaycastedLabels', 'PathToTrackingStream', 'PathToSavePcd', 'Value1', 'Value2', 'Value3',
                    'Value4', 'Value5', 'Value6']

    for spine_id in spine_ids:
        spine_id = f"sub-{spine_id}"

        extract_pcd_from_labelmaps_batch_file = os.path.join(args.root_path_spines, spine_id, "extract_pcd_from_labelmaps.txt")
        with open(extract_pcd_from_labelmaps_batch_file, 'w') as file:
            file.write(f'{"; ".join(placeholders)}\n')

        for deform in range(int(args.nr_deform_per_spine)):
            try:
                process_deformed_spine(args, deform, extract_pcd_from_labelmaps_batch_file, placeholders, spine_id)
            except Exception as e:
                print(f"failed to extract PCD from {spine_id} for deformation {deform} with error {e}")

        arguments_imfusion = f"batch={extract_pcd_from_labelmaps_batch_file}"
        print('ARGUMENTS: ', arguments_imfusion)
        os.system("ImFusionConsole" + " " + args.workspace_file + " " + arguments_imfusion)
        print('################################################### ')


def process_deformed_spine(args, deform, extract_pcd_from_labelmaps_batch_file, placeholders, spine_id):
    simulation_file = [
        file_name
        for file_name in os.listdir(os.path.join(args.root_path_spines, spine_id))
        if file_name.endswith('.imf') and 'us_sim' in file_name and f"field{deform}" in file_name
    ]
    if len(simulation_file) != 1:
        raise Exception(f"no ultrasound simulation found in {os.path.join(args.root_path_spines, spine_id)} for field")
    tracking_path = os.path.join(os.path.join(args.root_path_spines, spine_id), simulation_file[0]).replace(".imf",
                                                                                                            "tracking.ts")
    # we extract one point cloud per vertebra
    for vertebra in range(20, 25):
        dir_name_spine = os.path.join(args.root_path_spines, spine_id)
        dir_name_vert = os.path.join(args.root_path_vertebrae, spine_id + "_verLev" + str(vertebra))

        # deformed spine
        path_to_raycasted = os.path.join(dir_name_spine, "labels_force" + str(deform), "raycasted")
        path_to_saved = os.path.join(dir_name_vert, spine_id + "_verLev" + str(vertebra) + "_forces" + str(
                    deform) + "_deformed_raycasted.pcd")

        print("Extracting pointcloud from ultrasound for: " + str(spine_id) + "and deformation" + str(
            deform) + " and vert: " + str(vertebra))
        create_imfusion_args(extract_pcd_from_labelmaps_batch_file, path_to_raycasted, path_to_saved,
                             placeholders, tracking_path, vertebra)

        # source spine
        path_to_raycasted = os.path.join(dir_name_spine, "labels_force" + str(deform), "source", "raycasted")
        path_to_saved = os.path.join(dir_name_vert, spine_id + "_verLev" + str(vertebra) + "_forces" + str(
            deform) + "_source_raycasted.pcd")

        create_imfusion_args(extract_pcd_from_labelmaps_batch_file, path_to_raycasted, path_to_saved,
                             placeholders, tracking_path, vertebra)


def create_imfusion_args(extract_pcd_from_labelmaps_batch_file, path_to_raycasted, path_to_saved, placeholders,
                         tracking_path, vertebra):
    values_to_remove = ['18', '19', '20', '21', '22', '23', '92']
    # the current level of the vertebra we keep, the rest of the labels we replace with 0
    l_number = vertebra - 19
    values_to_remove.remove(str(23 - l_number))
    # create arguments list to call ImFusion with
    arguments = ""
    for p in placeholders:
        if p == 'PathToRaycastedLabels':
            value = path_to_raycasted
        elif p == 'PathToTrackingStream':
            value = tracking_path
        elif p == 'PathToSavePcd':
            value = path_to_saved
        elif p == 'Value1':
            value = values_to_remove[0]
        elif p == 'Value2':
            value = values_to_remove[1]
        elif p == 'Value3':
            value = values_to_remove[2]
        elif p == 'Value4':
            value = values_to_remove[3]
        elif p == 'Value5':
            value = values_to_remove[4]
        elif p == 'Value6':
            value = values_to_remove[5]
        arguments += str(value) + "; "
    with open(extract_pcd_from_labelmaps_batch_file, 'a') as file:
        parts = arguments.rsplit(";", 1)
        arguments = parts[0]
        file.write(f'{arguments}\n')


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

    process(args)
