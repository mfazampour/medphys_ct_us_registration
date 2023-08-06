import argparse
import os
from PIL import Image
import numpy as np
import cv2
import glob
import sys
import shutil


# Credits to Jane at https://github.com/lameski123/thesis/blob/main/processing_scripts/ray_cast_bmode_data.py
# faster ray-casting implemented

def ray_cast_image_w_color(image, bone_labels):
    rays = np.zeros_like(np.squeeze(np.squeeze(image)))

    marked = np.zeros(image.shape[1], dtype=np.int8)

    for i in range(image.shape[0]):
        bone_index = np.isin(image[i, :], bone_labels)
        to_set = np.where(bone_index, image[i, :], 0)
        to_set[marked != 0] = 0
        marked[to_set != 0] = 1
        rays[i, :] = to_set
        if marked.all():
            break

    return rays


def pointcloud_to_resliced_labelmap(subfolder, nr_deform, output_folder, workspace_file, read_source=False):

    simulation_file = [
        file_name
        for file_name in os.listdir(subfolder)
        if file_name.endswith('.imf') and 'us_sim' in file_name and f"field{nr_deform}" in file_name
    ]

    if len(simulation_file) != 1:
        raise Exception(f"no ultrasound simulation found in {subfolder} for field {nr_deform}")

    if read_source:
        label_map = [
            file_name
            for file_name in os.listdir(subfolder)
            if file_name.endswith('.nii.gz') and 'deformed' not in file_name and "seg" in file_name and "sim" not in file_name
        ]
        if len(label_map) != 1:
            raise Exception(f"no source labelmap found in {subfolder}, {label_map}")
    else:
        label_map = [
            file_name
            for file_name in os.listdir(subfolder)
            if file_name.endswith('.nii.gz') and 'deformed_seg' in file_name and f"field{nr_deform}" in file_name and "sim" not in file_name
        ]

        if len(label_map) != 1:
            raise Exception(f"no deformed labelmap found in {subfolder} for field {nr_deform}")

    placeholders = ['SIMULATEDSWEEP', 'DEFORMEDLABELMAP', 'OUTPUTIMAGEFOLDER', 'OUTPUTTRACKING']

    tracking_path = os.path.join(subfolder, simulation_file[0]).replace(".imf", "tracking.ts")
    arguments_imfusion = {placeholders[0]: os.path.join(subfolder, simulation_file[0]),
                          placeholders[1]: os.path.join(subfolder, label_map[0]),
                          placeholders[2]: output_folder,
                          placeholders[3]: tracking_path}
    arguments_imfusion = ' '.join([f"{key}={value}" for key, value in arguments_imfusion.items()])

    print('ARGUMENTS: ', arguments_imfusion)
    os.system("ImFusionConsole" + " " + workspace_file + " " + arguments_imfusion)
    print('################################################### ')

    return tracking_path


def process(args):
    # iterate over the txt file
    with open(args.txt_file) as file:
        spine_ids = [f"sub-{line.strip()}" for line in file]
    for spine_id in spine_ids:

        label_to_sweeo_batch_file = os.path.join(args.root_path_spines, spine_id, "label_to_sweep.txt")
        with open(label_to_sweeo_batch_file, 'w') as file:
            file.write('IMAGESET; INPUTTS; OUTPUTSWEEP\n')

        for deform in range(int(args.nr_deform_per_spine)):
            try:
                raycast_spine(args, deform, label_to_sweeo_batch_file, spine_id, read_source=True)
                raycast_spine(args, deform, label_to_sweeo_batch_file, spine_id, read_source=False)
            except Exception as e:
                print(f"Can not run raycast for {spine_id} and deform {deform}, error: {e}")
                continue
                # raycast_deformed_spine(args, deform, label_to_sweeo_batch_file, spine_id)

        workspace_file = "./imfusion_workspaces/imageset_to_sweep.iws"
        arguments_imfusion = f"batch={label_to_sweeo_batch_file}"
        print('ARGUMENTS: ', arguments_imfusion)
        os.system("ImFusionConsole" + " " + workspace_file + " " + arguments_imfusion)
        print('################################################### ')


def raycast_spine(args, deform, label_to_sweeo_batch_file, spine_id, read_source=False):
    if read_source:
        dir_save_path_full = os.path.join(args.root_path_spines, spine_id, "labels_force" + str(deform), "source", "full")
        dir_save_path = os.path.join(args.root_path_spines, spine_id, "labels_force" + str(deform), "source", "raycasted")
    else:
        dir_save_path_full = os.path.join(args.root_path_spines, spine_id, "labels_force" + str(deform), "full")
        dir_save_path = os.path.join(args.root_path_spines, spine_id, "labels_force" + str(deform), "raycasted")

    if os.path.exists(dir_save_path_full):
        shutil.rmtree(dir_save_path_full)
    os.makedirs(dir_save_path_full)
    if os.path.exists(dir_save_path):
        shutil.rmtree(dir_save_path)
    os.makedirs(dir_save_path)

    dir_name = os.path.join(args.root_path_spines, spine_id)

    # call imfusion to store the slice the whole spine point cloud according to the 2D image set

    tracking_path = pointcloud_to_resliced_labelmap(dir_name, deform, dir_save_path_full,
                                                "./imfusion_workspaces/labelmap_reslicing.iws", read_source=read_source)

    print("Raycasting: " + str(spine_id) + " and deformation " + str(deform))
    raycast_image_set(dir_save_path, dir_save_path_full, spine_id)

    if read_source:
        label_sweep_path = os.path.join(args.root_path_spines, spine_id, "labels_force" + str(deform),
                                        "source_raycasted_sweep.imf")
    else:
        label_sweep_path = os.path.join(args.root_path_spines, spine_id, "labels_force" + str(deform),
                                        "raycasted_sweep.imf")

    # add to batch file for creating a sweep from labels
    argument = f"{dir_save_path}; {tracking_path}; {label_sweep_path}"
    with open(label_to_sweeo_batch_file, 'a') as file:
        file.write(f'{argument}\n')


# def raycast_deformed_spine(args, deform, label_to_sweeo_batch_file, spine_id):
#     dir_save_path_full = os.path.join(args.root_path_spines, spine_id, "labels_force" + str(deform), "full")
#     if os.path.exists(dir_save_path_full):
#         shutil.rmtree(dir_save_path_full)
#     os.makedirs(dir_save_path_full)
#     dir_save_path = os.path.join(args.root_path_spines, spine_id, "labels_force" + str(deform), "raycasted")
#     if os.path.exists(dir_save_path):
#         shutil.rmtree(dir_save_path)
#     os.makedirs(dir_save_path)
#     dir_name = os.path.join(args.root_path_spines, spine_id)
#     # call imfusion to store the slice the whole spine point cloud according to the 2D image set
#     tracking_path = pointcloud_to_resliced_labelmap(dir_name, deform, dir_save_path_full,
#                                                     "./imfusion_workspaces/labelmap_reslicing.iws")
#     print("Raycasting: " + str(spine_id) + " and deformation " + str(deform))
#     raycast_image_set(dir_save_path, dir_save_path_full, spine_id)
#     label_sweep_path = os.path.join(args.root_path_spines, spine_id, "labels_force" + str(deform),
#                                     "raycasted_sweep.imf")
#     argument = f"{dir_save_path}; {tracking_path}; {label_sweep_path}"
#     with open(label_to_sweeo_batch_file, 'a') as file:
#         file.write(f'{argument}\n')


def raycast_image_set(dir_save_path, dir_save_path_full, spine_id):
    look_for = "**/*" + '*Images*' + '.png'
    filenames_labels = sorted(glob.glob(os.path.join(dir_save_path_full, look_for), recursive=True))
    if (len(filenames_labels) == 0):
        raise Exception("No labelmaps where found for spine: " + str(spine_id))
    bone_labels = [92, 18, 19, 20, 21, 22, 23]
    for filename_label in filenames_labels:
        label = np.array(Image.open(filename_label))
        ray_casted_labels = ray_cast_image_w_color(label, bone_labels)

        ray_casted_labels = cv2.dilate(ray_casted_labels, np.ones((3, 3)), iterations=1)

        label_name = os.path.basename(filename_label)
        image_save_path = os.path.join(dir_save_path, label_name.replace('Images', 'Images_raycasted'))

        Image.fromarray(ray_casted_labels).save(image_save_path)


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

    process(args)


