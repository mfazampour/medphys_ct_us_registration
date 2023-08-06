import argparse

import numpy as np
import torch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import Compose, RandRotate90, ToTensor, Activations, AsDiscrete
from monai.data import CacheDataset, DataLoader
from monai.utils import set_determinism
import wandb

from dataset import create_dataset

# Set random seed for reproducibility
torch.manual_seed(0)
set_determinism(seed=0)


def train(root_path_spine, txt_file):
    # Initialize WandB
    wandb.init(project="spine_sim_ultrasound_seg")

    batch_size = 250
    num_epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, val_dataset = create_dataset(root_path_spine, txt_file)

    # Create the data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    # Create the U-Net model
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(4, 8, 16, 32, 64, 128),
        strides=(2, 2, 2, 2, 2),
        bias=False,
    )

    # Move the model to the device
    model = model.to(device)

    # Define the loss function and metrics
    loss_function = DiceLoss(sigmoid=True)
    metric = DiceMetric(include_background=True, reduction="mean")
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold_values=True)])

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    # Initialize best validation metric
    best_val_metric = float('-inf')

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        train_loss = 0.0
        train_metric = 0.0
        val_loss = 0.0
        val_metric = 0.0

        total_batches = len(train_loader)
        for batch_idx, batch_data in enumerate(train_loader):
            inputs = batch_data["image"]['data'].to(device).squeeze(dim=4)
            labels = batch_data["label"]['data'].to(device).squeeze(dim=4)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # only calculate loss on non-zero labels
            indices = labels.sum(dim=(1, 2, 3)) != 0
            loss = loss_function(outputs[indices, ...], labels[indices, ...])

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Compute the training metrics
            train_loss += loss.item()
            metric(post_trans(outputs), labels)
            train_metric += metric.aggregate().item()

            # Print progress
            progress = (batch_idx + 1) / total_batches * 100
            print(f"\rProgress: {progress:.2f}% train metric: {train_metric / (batch_idx + 1)}", end="")

        # Compute the average training loss and metric
        train_loss /= len(train_loader)
        train_metric /= len(train_loader)

        # Evaluate on the validation set
        model.eval()
        with torch.no_grad():
            num_validation = np.min((len(val_loader), 50))  # limit the number of validations
            visualization_idx = np.random.choice(num_validation, 1)
            for idx, val_data in enumerate(val_loader):
                val_inputs = val_data["image"]['data'].to(device).squeeze(dim=4)
                val_labels = val_data["label"]['data'].to(device).squeeze(dim=4)
                val_outputs = model(val_inputs)
                val_loss += loss_function(val_outputs, val_labels).item()
                metric(post_trans(val_outputs), val_labels)
                val_metric += metric.aggregate().item()

                # Log one image with prediction and ground truth
                if idx == visualization_idx:
                    val_inputs_np = val_inputs.cpu().squeeze().numpy().T
                    val_labels_np = val_labels.cpu().squeeze().numpy().T
                    val_outputs_np = post_trans(val_outputs).squeeze().cpu().numpy().T
                    wandb.log({"Validation Image": wandb.Image(val_inputs_np),
                               "Ground Truth Label": wandb.Image(val_labels_np),
                               "Prediction": wandb.Image(val_outputs_np)})
                if idx == num_validation:
                    break

            val_loss /= num_validation
            val_metric /= num_validation

        print(f"\ntrain_loss: {train_loss}, val_loss: {val_loss}, train_metric: {train_metric}, val_metric: {val_metric}")
        # Log the losses to WandB
        wandb.log(
            {"train_loss": train_loss, "val_loss": val_loss, "train_metric": train_metric, "val_metric": val_metric})

        # Save the best model
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            torch.save(model.state_dict(), "/mnt/HDD1/farid/spine_verse/checkpoints/best_seg_model.pt")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="replace the labels with ultrasound simuation labels")

    arg_parser.add_argument(
        "--root_path_spine",
        required=True,
        dest="root_path_spine",
        help="Root path of the spine folders with cropped spines"
    )

    arg_parser.add_argument(
        "--list_file_names",
        required=True,
        dest="txt_file",
        help="Txt file that contains all spines that contain all lumbar vertebrae"
    )

    args = arg_parser.parse_args()
    train(args.root_path_spine, args.txt_file)