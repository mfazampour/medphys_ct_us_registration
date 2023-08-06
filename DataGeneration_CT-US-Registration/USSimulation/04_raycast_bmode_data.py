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
        spine_ids = [line.strip() for line in file]

    for spine_id in spine_ids:
        for deform in range(int(args.nr_deform_per_spine)):

            dir_save_path = os.path.join(args.root_path_spines,spine_id, "labels_force" + str(deform), "raycasted")
            if os.path.exists(dir_save_path):
                shutil.rmtree(dir_save_path)
            os.makedirs(dir_save_path)

            print("Raycasting: " + str(spine_id) + " and deformation " + str(deform) )
            look_for = "**/*" + '*Images*' + '.png'
            filenames_labels = sorted(glob.glob(os.path.join(args.root_path_spines, spine_id, "labels_force" + str(deform), look_for), recursive=True))

            if (len(filenames_labels) == 0):
                print("No labelmaps where found for spine: " + str(spine_id), file=sys.stderr)
                continue

            for filename_label in filenames_labels:
                label = np.array(Image.open(filename_label))
                ray_casted_labels = ray_cast_image(label)

                ray_casted_labels = cv2.dilate(ray_casted_labels, np.ones((3, 3)), iterations=1)

                label_name = os.path.basename(filename_label)
                image_save_path = os.path.join(dir_save_path, label_name.replace('.png','_raycasted.png'))

                Image.fromarray(ray_casted_labels).save(image_save_path)

