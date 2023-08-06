import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

os.environ['PATH'] = 'C:\\Program Files\\ImFusion\\ImFusion Suite\\Suite;C:\\Program Files\\ImFusion' \
                     '\\ImFusion Suite\\Suite\\plugins;' + os.environ['PATH']
# PYTHONUNBUFFERED = 1;PYTHONPATH=C:\Program Files\ImFusion\ImFusion Suite\Suite\;
import imfusion


def get_image_points(image, T, image_width, image_height):
    """
    :param image_width: image physical width
    :param image_height: image physical height
    """

    # binary_image = np.where(image>0.5, 1, 0)
    # plt.subplot(1, 2, 1)
    # plt.imshow(image)
    # plt.subplot(1, 2, 2)
    # plt.imshow(binary_image)
    # plt.show()

    image = np.transpose(image)

    image_spacing = np.array([image_width/image.shape[0], image_height/image.shape[1]])
    image_spacing = np.expand_dims(image_spacing, 0)

    img_physical_size = np.array([image_width, image_height])

    idx_apex = np.argwhere(image > 0.5)
    points_apex = idx_apex * image_spacing
    points_center = points_apex - img_physical_size/2

    # making point homogenous and adding axis 0 = 0
    points_center = np.concatenate( (points_center,
                                     np.zeros( (points_center.shape[0], 1)),
                                     np.ones( (points_center.shape[0], 1))), axis=1  )

    points_center = np.transpose(points_center)

    points_word = np.matmul(T, points_center)
    return np.transpose(points_word)[:, 0:3]


def get_point_cloud(data_list, transform_list, image_width, image_height):
    """
    :param image_width: image physical width
    :param image_height: image physical height
    """

    pc_list = []
    for T, image_path in zip(transform_list, data_list):
        image = np.squeeze(np.load(image_path))

        pc_list.append(get_image_points(image, T, image_width, image_height))

    overall_pc = np.concatenate(pc_list, axis=0)

    return overall_pc


def get_id_number_from_image_name(image_name):
    # e.g. spine10_ts_1_0_022.npz -> 22

    # 1. spine10_ts_1_0_022.npz -> "022"
    image_number = image_name.split("_")[-2].split(".")[0]
    return int(image_number)


def reorder_ts_data(ts_path):
    image_file_names = [item for item in os.listdir(ts_path) if "pred" in item]
    image_numbers = [get_id_number_from_image_name(item) for item in image_file_names]

    sorted_idxes = sorted(range(len(image_numbers)), key=image_numbers.__getitem__)
    image_file_names = [os.path.join(ts_path, image_file_names[idx]) for idx in sorted_idxes ]

    return image_file_names


def get_us_meta_info(us_sweep):

    meta_info = {'spacing': us_sweep[0].spacing,
                 'physical_width': us_sweep[0].width * us_sweep[0].spacing[0],
                 'physical_height': us_sweep[0].height * us_sweep[0].spacing[1],
                 'transform_list': [us_sweep.matrix(i) for i in range(1, len(us_sweep))]}

    return meta_info


def main(unet_segmentation_path, us_sweep_dir, save_dir):

    # grouping the images that belongs to the same timestamp
    for spine_id in os.listdir(unet_segmentation_path):
        if "spine" not in spine_id:
            continue
        ts_list = [item for item in os.listdir(os.path.join(unet_segmentation_path, spine_id))]

        for ts_name in ts_list:

            us_filepath = os.path.join(us_sweep_dir, spine_id, "ts_" + ts_name) + ".imf"
            if not os.path.exists(us_filepath):
                print("non existing file: ", us_filepath)
                continue

            us_sweep, = imfusion.open( os.path.join(us_sweep_dir, spine_id, "ts_" + ts_name) + ".imf")
            meta_data = get_us_meta_info(us_sweep)

            ordered_filenames = reorder_ts_data(os.path.join(unet_segmentation_path, spine_id, ts_name))
            us_point_cloud = get_point_cloud(data_list = ordered_filenames,
                                             transform_list=meta_data['transform_list'],
                                             image_width=meta_data['physical_width'],
                                             image_height=meta_data['physical_height'], )

            save_folder = os.path.join(save_dir, spine_id)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            np.savetxt(os.path.join(save_folder, "raycasted_ts_" + ts_name + ".txt"), us_point_cloud)

            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data generation testing')
    parser.add_argument('--dataset_path', type=str, default="E:/NAS/jane_project/unet_us_segmentation")
    parser.add_argument('--us_sweep_dir', type=str, default="E:/NAS/jane_project/simulated_us")
    parser.add_argument('--save_path', type=str, default="E:/NAS/jane_project/us_segmented_point_clouds")

    args = parser.parse_args()

    imfusion.init()

    main(args.dataset_path, args.us_sweep_dir, args.save_path)
