from data import SceneflowDataset
import argparse
import numpy as np
import os


def main(dataset_path, save_dir):
    test_set = SceneflowDataset(mode="train", root=dataset_path, raycasted=True)

    results = []
    for i, data in enumerate(test_set):
        source_pc, target_pc, color1, color2, gt_flow, mask1, constraint, position1, position2, file_name, tre_points \
            = data

        save_folder = os.path.join(save_dir, file_name)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        np.savetxt(os.path.join(save_folder, "source_pc.txt"), source_pc[:, 0:3])
        np.savetxt(os.path.join(save_folder, "target_pc.txt"), target_pc[:, 0:3])
        np.savetxt(os.path.join(save_folder, "gt_predicted.txt"), source_pc[:, 0:3] + gt_flow)
        np.savetxt(os.path.join(save_folder, "tre_points.txt"), tre_points[:, 0:3])
        np.savetxt(os.path.join(save_folder, "constraints.txt"), source_pc[constraint, 0:3])


main(dataset_path="E:/NAS/jane_project/flownet_data/nas_data/new_data_raycasted",
     save_dir="C:/Users/maria/OneDrive/Desktop/data_check")
