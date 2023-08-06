import numpy as np
import wandb


def plot_pointcloud(flow_pred, pc1, pc2, tag='', mode='training'):
    pc1 = pc1.transpose(1, 2).detach().cpu().numpy()[0, :, :].squeeze()
    pc2 = pc2.transpose(1, 2).detach().cpu().numpy()[0, :, :].squeeze()
    flow_pred = flow_pred.transpose(1, 2).detach().cpu().numpy()[0, :, :].squeeze()
    to_plot = np.zeros((pc1.shape[0] * 3, 6))
    to_plot[:pc1.shape[0], :3] = pc1[:, :3]
    to_plot[:pc1.shape[0], 3] = 255  # red -> pc1 will be red
    to_plot[pc1.shape[0]:pc1.shape[0] * 2, :3] = pc1[:, :3] + flow_pred
    to_plot[pc1.shape[0]:pc1.shape[0] * 2, 4] = 255  # green -> pc1 + flow will be green
    to_plot[pc1.shape[0] * 2:, :3] = pc2[:, :3]
    to_plot[pc1.shape[0] * 2:, 5] = 255  # blue -> pc2 will be blue
    tag = mode if tag == '' else f"{mode}_{tag}"
    wandb.log({
        tag: wandb.Object3D({"type": "lidar/beta", "points": to_plot})
    })
