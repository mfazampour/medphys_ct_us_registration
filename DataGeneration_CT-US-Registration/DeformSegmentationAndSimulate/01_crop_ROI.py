import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchio
import torchio as tio
import matplotlib.pyplot as plt
from torchio.transforms import Crop, Pad, Resample, ToCanonical


def process(txt_file, root_path_spine, segmentation_folder, out_path_spine):
    # Read the lines from the text file
    with open(txt_file, 'r') as file:
        lines = file.readlines()

    # Process each line
    for line in lines:
        line = line.strip()  # Remove leading/trailing whitespaces
        subfolder_path = os.path.join(root_path_spine, line)
        output_folder = os.path.join(out_path_spine, f"sub-{line}")

        # Find the nii.gz file without 'seg' in its name
        nii_files = [
            file_name
            for file_name in os.listdir(subfolder_path)
            if file_name.endswith('.nii.gz') and 'seg' not in file_name
        ]

        if len(nii_files) > 0:
            print(f"cropping image for data {line}")
            # Open the first nii.gz file found using torchio
            nii_path = os.path.join(subfolder_path, nii_files[0])
            image = tio.ScalarImage(nii_path)
            # Do something with the image, e.g., apply transformations or print information
            print(f"Loaded image: {nii_path}")

            # Construct the segmentation file path
            segmentation_file = f"{line}_segmentation.nrrd"
            segmentation_path = os.path.join(segmentation_folder, segmentation_file)

            # Open the segmentation file using torchio
            segmentation = tio.LabelMap(segmentation_path)

            # change to LAS if not already
            if "".join(image.orientation) != 'LAS':
                print(f"image of {line} is not in canonical form. transforming it")
                t = ToCanonical()
                image = t(image)
                segmentation = t(segmentation)

            if image.shape != segmentation.shape:  # there is a chance that output of totalsegmentator is slightly bigger
                t = Resample(image)
                segmentation = t(segmentation)

            cropped_img, cropped_seg = crop(image, segmentation)
            cropped_seg = segment_soft_tissue(cropped_img, cropped_seg)
            cropped_img.save(os.path.join(output_folder, f"{line}_cropped.nii.gz"), squeeze=True)
            cropped_seg.save(os.path.join(output_folder, f"{line}_segmentation_cropped.nii.gz"), squeeze=True)
        else:
            print(f"No matching nii.gz files found in {subfolder_path}")


def segment_soft_tissue(image, segmentation: torchio.Image):
    seg_data = segmentation.data
    seg_data = region_growing(image, segmentation, threshold=[-100, 200], region_label=8)
    # seg_data[(image.data > 0) & (image.data < 200) & (segmentation.data == 0)] = 109  # muscle
    seg_data[(image.data > -300) & (image.data <= 200) & (seg_data == 0)] = 3  # fat

    segmentation.set_data(seg_data)
    return segmentation


def crop(image: torchio.Image, segmentation: torchio.Image):
    coronal_crop_index, end_coronal_index = mask_minimum_lateral_index(image, segmentation)
    (start_sag, end_sag, lowest_sacrum_index, lowest_t1_index) = mask_middle_part(image, segmentation)

    t = Crop([start_sag, image.shape[1] - end_sag,
              coronal_crop_index, image.shape[2] - end_coronal_index,
              lowest_sacrum_index, image.shape[3] - lowest_t1_index])

    image = t(image)
    segmentation = t(segmentation)
    pad_size = int(50 // image.spacing[1])  # pad 50 mm so that the probe is still inside the volume
    t = Pad([0, 0, pad_size, 0, 0, 0], padding_mode="minimum")
    image = t(image)
    segmentation = t(segmentation)

    # visualize(image)
    return image, segmentation


def dilate_labels(label_map: torch.Tensor, kernel_size):
    # Create a dilating kernel
    kernel = torch.ones((1, 1, kernel_size, kernel_size, kernel_size), dtype=torch.float32).to(label_map.device)

    # Pad the label map to handle border pixels
    padding = (kernel_size - 1) // 2
    padded_label_map = F.pad(label_map.unsqueeze(0).unsqueeze(0).float(), (padding, padding, padding, padding, padding, padding), mode='constant', value=0)

    # Perform 3D dilation using convolution
    dilated_label_map = F.conv3d(padded_label_map, kernel, padding=0)

    return dilated_label_map.squeeze()


def region_growing(image: torchio.Image, segmentation: torchio.Image, threshold, region_label=8):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    image_data = image.data.squeeze().to(device)

    seg_data = segmentation.data.squeeze().to(device)
    seg_data[(seg_data == 95) | (seg_data == 94) | (seg_data == 100) |
             (seg_data == 101) | (seg_data == 109)] = 8  # muscle

    kernel_size = 3

    region_mask = (seg_data == region_label)

    sum_filtered = None

    rep = 0

    while rep < 5:
        dilated = dilate_labels(region_mask, kernel_size)
        filtered = torch.where((dilated != 0) & (seg_data == 0) & (threshold[0] < image_data) & (threshold[1] >= image_data), 1, 0)  # & (threshold[0] < image_data) & (threshold[1] >= image_data)
        if filtered.sum() == 0:
            break
        seg_data[filtered == 1] = region_label
        region_mask = (seg_data == region_label)
        rep += 1

    return seg_data.unsqueeze(0).cpu()


def region_growing_1(image: torchio.Image, segmentation: torchio.Image, threshold, region_label=8):

    image_data = image.data.squeeze()

    seg_data = segmentation.data.squeeze()
    seg_data[(seg_data == 95) | (seg_data == 94) | (seg_data == 100) |
             (seg_data == 101) | (seg_data == 109)] = 8  # muscle

    # Get the labeled region
    region_mask = (seg_data == region_label)

    border_mask = (region_mask ^ torch.roll(region_mask, 1, dims=0)) | (region_mask ^ torch.roll(region_mask, -1, dims=0)) | \
                  (region_mask ^ torch.roll(region_mask, 1, dims=1)) | (region_mask ^ torch.roll(region_mask, -1, dims=1)) | \
                  (region_mask ^ torch.roll(region_mask, 1, dims=2)) | (region_mask ^ torch.roll(region_mask, -1, dims=2))

    border_coords = torch.argwhere(border_mask & (seg_data == 0))  # only choose the ones that have background value

    # Create an array to keep track of visited pixels
    visited = torch.zeros_like(seg_data, dtype=torch.bool).to(image_data.device)

    # Initialize the region
    region = torch.zeros_like(seg_data, dtype=bool).to(image_data.device)

    # Perform region growing
    queue = list(border_coords)
    while queue:
        # Get the next pixel from the queue
        pixel = queue.pop(0)

        # Mark the pixel as visited
        visited[pixel[0], pixel[1], pixel[2]] = True

        # Check if the pixel intensity is within the threshold
        if (threshold[0] < image_data[pixel[0], pixel[1], pixel[2]] <= threshold[1]) \
                and seg_data[pixel[0], pixel[1], pixel[2]] == 0:
            # Add the pixel to the region
            region[pixel[0], pixel[1], pixel[2]] = True

            # Get the neighbors of the pixel
            neighbors = [
                (pixel[0] - 1, pixel[1], pixel[2]),
                (pixel[0] + 1, pixel[1], pixel[2]),
                (pixel[0], pixel[1] - 1, pixel[2]),
                (pixel[0], pixel[1] + 1, pixel[2]),
                (pixel[0], pixel[1], pixel[2] - 1),
                (pixel[0], pixel[1], pixel[2] + 1)
            ]

            # Add the unvisited neighbors to the queue
            for neighbor in neighbors:
                if (
                        0 <= neighbor[0] < image_data.shape[0] and
                        0 <= neighbor[1] < image_data.shape[1] and
                        0 <= neighbor[2] < image_data.shape[2] and
                        not visited[neighbor[0], neighbor[1], neighbor[2]]
                ):
                    queue.append(neighbor)

    return region


def mask_middle_part(image, segmentation):
    _, H, W, D = image.shape
    start_sag = H // 4
    end_sag = 3 * H // 4

    center = H // 2
    if (center - start_sag) * image.spacing[
        0] < 100:  # if distance is less than 100 mm take at least 100m from each side
        print("volume is too thin, need bigger slab")
        start_sag = int(np.max([center - 100 // image.spacing[0], 0]))
        end_sag = int(np.min([center + 100 // image.spacing[0], H - 1]))

    # mask in the Z direction
    segmentation_data = segmentation.data.squeeze()
    # find sacrum the lowest point
    lowest_sacrum_index = 0
    for i in range(segmentation_data.shape[2]):
        slice_ = segmentation_data[:, :, i]
        if (slice_ == 92).sum() > 0:
            lowest_sacrum_index = i
            break
    # find T11 lowest point
    lowest_t1_index = segmentation_data.shape[2] - 1
    if lowest_sacrum_index is not None:
        for i in range(lowest_sacrum_index, segmentation_data.shape[2]):
            slice_ = segmentation_data[:, :, i]
            if (slice_ == 24).sum() > 0:
                lowest_t1_index = i
                break

    return (start_sag, end_sag, lowest_sacrum_index, lowest_t1_index)


def mask_minimum_lateral_index(image, segmentation):
    # Get the data from the segmentation label
    segmentation_data = segmentation.data.squeeze()

    bone_data = torch.where(segmentation_data == 92 | ((segmentation_data >= 18) & (segmentation_data < 27)), 1.0, 0.0)

    # Find the lowest index in the coronal plane that still shows bone
    lowest_coronal_index = None
    for i in range(bone_data.shape[1]):
        coronal_slice = bone_data[:, i, :]
        if coronal_slice.sum() > 0:
            lowest_coronal_index = i
            break

    image_data = image.data.squeeze()

    # find where we have something in the segmentation method, get the mean position
    indices = (segmentation_data[:, lowest_coronal_index, :] > 0).nonzero(as_tuple=True)
    center = [indices[0].median(), indices[1].median()]

    # then scroll down from the position until you find where there is air and cut from there
    coronal_crop_index = 0
    for i in range(lowest_coronal_index, 0, -1):
        coronal_value = image_data[center[0], i, center[1]]
        if coronal_value < -700:
            # we reached air, let's crop here
            coronal_crop_index = i
            break

    end_coronal_index = segmentation_data.shape[1]
    for i in range(image_data.shape[1] - 1, 0, -1):
        coronal_slice = image_data[:, i, :]
        if (coronal_slice > -500).any():
            end_coronal_index = i
            break

    return coronal_crop_index, end_coronal_index


def visualize(image):
    # Get the image data
    image_data = image.data.squeeze()

    # Get the center index along each axis
    center_x = image_data.shape[0] // 2
    center_y = image_data.shape[1] // 2
    center_z = image_data.shape[2] // 2

    # Extract the three central slices in each direction
    coronal_slice = image_data[center_x, :, :]
    sagittal_slice = image_data[:, center_y, :]
    axial_slice = image_data[:, :, center_z]

    # Visualize the slices
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    # Axial slices
    axes[0, 0].imshow(axial_slice, cmap='gray')
    axes[0, 0].set_title('Axial Slice (z-axis)')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(image_data[:, :, center_z - 1], cmap='gray')
    axes[0, 1].set_title('Axial Slice (z-axis - 1)')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(image_data[:, :, center_z + 1], cmap='gray')
    axes[0, 2].set_title('Axial Slice (z-axis + 1)')
    axes[0, 2].axis('off')

    # Sagittal slices
    axes[1, 0].imshow(sagittal_slice, cmap='gray')
    axes[1, 0].set_title('Sagittal Slice (y-axis)')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(image_data[:, center_y - 1, :], cmap='gray')
    axes[1, 1].set_title('Sagittal Slice (y-axis - 1)')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(image_data[:, center_y + 1, :], cmap='gray')
    axes[1, 2].set_title('Sagittal Slice (y-axis + 1)')
    axes[1, 2].axis('off')

    # Coronal slices
    axes[2, 0].imshow(coronal_slice, cmap='gray')
    axes[2, 0].set_title('Coronal Slice (x-axis)')
    axes[2, 0].axis('off')

    axes[2, 1].imshow(image_data[center_x - 1, :, :], cmap='gray')
    axes[2, 1].set_title('Coronal Slice (x-axis - 1)')
    axes[2, 1].axis('off')

    axes[2, 2].imshow(image_data[center_x + 1, :, :], cmap='gray')
    axes[2, 2].set_title('Coronal Slice (x-axis + 1)')
    axes[2, 2].axis('off')

    plt.tight_layout()
    plt.show()
    pass


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Crop the CT and fill the segmentation volume with soft tissue")

    arg_parser.add_argument(
        "--root_path_spine",
        required=True,
        dest="root_path_spine",
        help="Root path of the spine folders"
    )

    arg_parser.add_argument(
        "--list_file_names",
        required=True,
        dest="txt_file",
        help="Txt file that contains all spines that contain all lumbar vertebrae"
    )

    arg_parser.add_argument(
        "--segmentation_folder",
        required=True,
        help="folder containing the output of the total segmentator"
    )

    arg_parser.add_argument(
        "--out_path_spine",
        required=True,
        help="output folder path for cropped spine folders"
    )

    # arg_parser.add_argument(
    #     "--workspace_scale_mesh",
    #     required=True,
    #     dest="workspace_scale_mesh",
    #     help="ImFusion workspace files that scales a mesh to 0.001 of its original size"
    # )

    args = arg_parser.parse_args()
    process(args.txt_file, args.root_path_spine, args.segmentation_folder, args.out_path_spine)
