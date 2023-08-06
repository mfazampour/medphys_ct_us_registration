import argparse
import os
from PIL import Image
import numpy as np
import cv2
import glob
import sys
import shutil


# Credits to Jane at https://github.com/lameski123/thesis/blob/main/processing_scripts/ray_cast_bmode_data.py

def ray_cast_image(image):
    rays = np.zeros_like(np.squeeze(np.squeeze(image)))
    j_range = range(image.shape[0])

    for i in range(image.shape[1]):
        for j in j_range:
            if image[j, i] != 0:
                rays[j, i] = 1
                break

    return rays


def pointcloud_to_resliced_labelmap(subfolder, nr_deform, output_folder, workspace_file):

    simulation_file = [
        file_name
        for file_name in os.listdir(subfolder)
        if file_name.endswith('.imf') and 'us_sim' in file_name and f"field{nr_deform}" in file_name
    ]

    if len(simulation_file) != 1:
        raise Exception(f"no ultrasound simulation found in {subfolder} for field {nr_deform}")

    deformed_pc = [
        file_name
        for file_name in os.listdir(subfolder)
        if file_name.endswith('.obj') and 'deformed' in file_name and f"field{nr_deform}" in file_name and "centered" not in file_name
    ]

    if len(deformed_pc) != 1:
        raise Exception(f"no deformed pc found in {subfolder} for field {nr_deform}")

    placeholders = ['SIMULATEDSWEEP', 'DEFORMEDPC', 'OUTPUTIMAGEFOLDER', 'OUTPUTTRACKING']

    tracking_path = os.path.join(subfolder, simulation_file[0]).replace(".imf", "tracking.ts")
    arguments_imfusion = {placeholders[0]: os.path.join(subfolder, simulation_file[0]),
                          placeholders[1]: os.path.join(subfolder, deformed_pc[0]),
                          placeholders[2]: output_folder,
                          placeholders[3]: tracking_path}
    arguments_imfusion = ' '.join([f"{key}={value}" for key, value in arguments_imfusion.items()])

    print('ARGUMENTS: ', arguments_imfusion)
    os.system("ImFusionConsole" + " " + workspace_file + " " + arguments_imfusion)
    print('################################################### ')

    return tracking_path


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Raycast the US labelmaps to resemble US segmentations")

    arg_parser.add_argument(
        "--list_file_names",
        required=True,
        dest="txt_file",
        help="Txt file that contains all spines that contain all lumbar vertebrae"
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
    print("Raycast the US labelmaps to resemble US segmentations")

    # iterate over the txt file
    with open(args.txt_file) as file:
        spine_ids = [f"sub-{line.strip()}" for line in file]


    for spine_id in spine_ids:

        label_to_sweeo_batch_file = os.path.join(args.root_path_spines, spine_id, "label_to_sweep.txt")
        with open(label_to_sweeo_batch_file, 'w') as file:
            file.write('IMAGESET; INPUTTS; OUTPUTSWEEP\n')

        for deform in range(int(args.nr_deform_per_spine)):

            dir_save_path_full = os.path.join(args.root_path_spines, spine_id, "labels_force" + str(deform), "full")
            if os.path.exists(dir_save_path_full):
                shutil.rmtree(dir_save_path_full)
            os.makedirs(dir_save_path_full)

            dir_save_path = os.path.join(args.root_path_spines, spine_id, "labels_force" + str(deform), "raycasted")
            if os.path.exists(dir_save_path):
                shutil.rmtree(dir_save_path)
            os.makedirs(dir_save_path)

            dir_name = os.path.join(args.root_path_spines, spine_id)
            # call imfusion to store the slice the whole spine point cloud according to the 2D image set
            tracking_path = pointcloud_to_resliced_labelmap(dir_name, deform, dir_save_path_full,
                                                            "./imfusion_workspaces/pointcloud_to_labelmap_reslicing.iws")

            print("Raycasting: " + str(spine_id) + " and deformation " + str(deform))
            look_for = "**/*" + '*Images*' + '.png'
            filenames_labels = sorted(glob.glob(os.path.join(dir_save_path_full, look_for), recursive=True))

            if (len(filenames_labels) == 0):
                print("No labelmaps where found for spine: " + str(spine_id), file=sys.stderr)
                continue

            for filename_label in filenames_labels:
                label = np.array(Image.open(filename_label))
                ray_casted_labels = ray_cast_image(label)

                ray_casted_labels = cv2.dilate(ray_casted_labels, np.ones((3, 3)), iterations=1)

                label_name = os.path.basename(filename_label)
                image_save_path = os.path.join(dir_save_path, label_name.replace('Images', 'Images_raycasted'))

                Image.fromarray(ray_casted_labels).save(image_save_path)

            label_sweep_path = os.path.join(args.root_path_spines, spine_id, "labels_force" + str(deform), "raycasted_sweep.imf")
            argument = f"{dir_save_path}; {tracking_path}; {label_sweep_path}"
            with open(label_to_sweeo_batch_file, 'a') as file:
                file.write(f'{argument}\n')

        workspace_file = "./imfusion_workspaces/imageset_to_sweep.iws"
        arguments_imfusion = f"batch={label_to_sweeo_batch_file}"
        print('ARGUMENTS: ', arguments_imfusion)
        os.system("ImFusionConsole" + " " + workspace_file + " " + arguments_imfusion)
        print('################################################### ')


