import argparse
import os
import shutil

import torch
from monai.networks.nets import UNet
from monai.transforms import Compose, ToTensor, ToNumpy, Activations, AsDiscrete
import torchio as tio

class SegmentationInference:

    def __init__(self, path_to_model):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Create the U-Net model
        self.model = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(4, 8, 16, 32, 64, 128),
            strides=(2, 2, 2, 2, 2),
            bias=False,
        )

        # Move the model to the device
        self.model = self.model.to(self.device)

        # Load the saved model state dictionary
        self.model.load_state_dict(torch.load(path_to_model))
        self.model.eval()

    def infer(self, image_path) -> tio.Image:

        # Define the transform for inference
        # transform = Compose([ToTensor(), ToNumpy()])
        img = tio.ScalarImage(image_path)

        t_forward = tio.Compose([tio.CropOrPad((512, 512, 1)),
                                  tio.RescaleIntensity(out_min_max=(0, 1)),
                                  ])

        t_backward = tio.Resample(img)

        # Apply the transform to the input image
        img_t = t_forward(img)
        input_tensor = img_t.data.permute((3, 0, 1, 2)).to(self.device)

        # # Add batch dimension to the input tensor
        # input_tensor = input_tensor.unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            output_tensor = self.model(input_tensor)
            post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold_values=True)])
            output_tensor = post_trans(output_tensor)

        # Convert the output tensor to a NumPy array
        img_t.set_data(output_tensor.permute((1, 2, 3, 0)).cpu())

        output_img = t_backward(img_t)

        return output_img


def find_vertebra_level(output_img: tio.Image, gt_label: str):
    label_data = tio.LabelMap(gt_label).data
    output_data = output_img.data
    if len(label_data.unique()) == 1:  # it has only zeros
        output_data *= 0.0
    elif len(label_data.unique()) == 2:  # it has only one value other than zero
        value = label_data.unique(sorted=True)[1]
        output_data[output_data != 0] = value
    else:
        idx_label = label_data.nonzero()
        idx_output = output_data.nonzero()
        dists = torch.cdist(idx_output.float(), idx_label.float())
        min_idx = torch.argmin(dists, dim=1)
        output_data[output_data.nonzero(as_tuple=True)] = label_data[:, idx_label[min_idx, :][:, 1],
                                                          idx_label[min_idx, :][:, 2], :].squeeze().float()
    output_img.set_data(output_data)
    return output_img


def process(root_path_spine, txt_file, model_path, nr_deform_per_spine):

    segmentator = SegmentationInference(model_path)

    # Read the lines from the text file
    with open(txt_file, 'r') as file:
        lines = file.readlines()

    # Process each line
    for line in lines:
        line = line.strip()  # Remove leading/trailing whitespaces
        subfolder_path = os.path.join(root_path_spine, f"sub-{line}/")

        print(f"processing patient {line}")

        for force in range(nr_deform_per_spine):
            ultrasound_folder = os.path.join(subfolder_path, f"sub-{line}forcefield{force}_us_set/")
            # check if folder is exists and if it is not empty
            if not os.path.exists(ultrasound_folder):
                continue
            if os.listdir(ultrasound_folder).__len__() == 0:
                continue

            target_folder = os.path.join(subfolder_path, f"labels_force{force}/", "net_us_seg/")
            if os.path.exists(target_folder):
                shutil.rmtree(target_folder)
            os.makedirs(target_folder)

            # Find the nii.gz file without 'seg' in its name
            png_files = [
                file_name
                for file_name in os.listdir(ultrasound_folder)
                if file_name.endswith('.png')
            ]

            print(f"inferring on {ultrasound_folder}, saving the result in {target_folder}")

            for file in png_files:
                label = segmentator.infer(os.path.join(ultrasound_folder, file))
                gt_label_path = os.path.join(subfolder_path, f"labels_force{force}/raycasted/", file.replace("Ultrasound", "Images_raycasted"))
                label = find_vertebra_level(label, gt_label_path)
                label.save(os.path.join(target_folder, file.replace("Ultrasound", "Ultrasound_label")))


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(description="Extract point clouds from raycasted ultrasound labelmaps")

    arg_parser.add_argument(
        "--list_file_names",
        required=True,
        dest="txt_file",
        help="Txt file that contains all spines that contain all lumbar vertebrae"
    )
    arg_parser.add_argument(
        "--model_path",
        required=True,
        help="path to the saved segmentation model"
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
        type=int,
        help="Number of deformations per spine"
    )

    args = arg_parser.parse_args()
    process(args.root_path_spines, args.txt_file, args.model_path, args.nr_deform_per_spine)
