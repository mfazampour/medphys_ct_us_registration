import itertools
import numpy as np
from scipy.spatial import cKDTree
import os
import matplotlib.pyplot as plt
import time

os.environ['PATH'] = 'C:\\Program Files\\ImFusion\\ImFusion Suite\\Suite;C:\\Program Files\\ImFusion' \
                     '\\ImFusion Suite\\Suite\\plugins;' + os.environ['PATH']

import imfusion


def make_homogeneous(pc):
    if pc.shape[0] != 3:
        pc = np.transpose(pc)

    assert pc.shape[0] == 3

    return np.concatenate((pc, np.ones((1, pc.shape[1]))), axis=0)


def voxels2pyhsicalspace(voxel_coordinates, spacing, physical_size, T_data2world):
    """
    :param: indexes_list: [Nx3] list of points in voxel coordinates
    :param: spacing: The voxel spacing
    :param physical_size: The volume physical size
    :param T_data2world: The volume transform in the world coordinate system

    :return: [Nx3] array containing the voxels coordinates in physical space
    """

    assert isinstance(voxel_coordinates, np.ndarray), "voxel_coordinates must be a Nx3 array"
    assert voxel_coordinates.shape[1] == 3

    spacing = np.array(spacing)

    # Points expressed in physical coordinates wrt to top left corner of the volume
    points_apex = np.multiply(voxel_coordinates, np.expand_dims(spacing, axis=0))

    # # Adding half voxel, assuming that we consider voxel centers - This is only needed for very large voxels,
    # # otherwise is negligible
    points_apex = np.add(points_apex, np.expand_dims(spacing / 2, axis=0))

    # Point expressed in physical coordinates wrt to the volume center
    points_vol_center = np.add(points_apex, -np.expand_dims(physical_size / 2, axis=0))

    points_vol_center = make_homogeneous(points_vol_center)
    physical_points = np.matmul(T_data2world, points_vol_center)

    return np.transpose(physical_points[0:3, ...])


def get_bounding_box(physical_points):
    """

    :param physical_points: [Nx3] array containing point coordinates
    :return: The bounding box containing the input physical_points
    """

    return [(np.min(physical_points[:, 0]), np.max(physical_points[:, 0])),
            (np.min(physical_points[:, 1]), np.max(physical_points[:, 1])),
            (np.min(physical_points[:, 2]), np.max(physical_points[:, 2]))]


def find_indexes_in_box(points, box):
    """
    :param points: [Nx3] array containing point coordinates
    :param box: list of tuples like [(box_x_min, box_x_max), (box_y_min, box_y_max), (box_z_min, box_z_max)]
    :return: The indexes of the points contained in the bounding box
    """

    x_in_box = np.logical_and(points[:, 0] >= box[0][0], points[:, 0] <= box[0][1])
    y_in_box = np.logical_and(points[:, 1] >= box[1][0], points[:, 1] <= box[1][1])
    z_in_box = np.logical_and(points[:, 2] >= box[2][0], points[:, 2] <= box[2][1])

    x_y_in_box = np.logical_and(x_in_box, y_in_box)
    indexes_in_box = np.logical_and(x_y_in_box, z_in_box)

    return np.argwhere(indexes_in_box).flatten()


def grid2physicalspace(grid_size, spacing, T_data2world):

    dims = len(grid_size)
    physical_size = np.array([grid_size[i] * spacing[i] for i in range(dims)])
    spacing = np.array(spacing)

    a = [ list(range(0, size)) for size in grid_size]

    # getting the indexes list with the first dimension changing slowelier and last one changing faster
    indexes_list = list(itertools.product(*a))
    physical_points = voxels2pyhsicalspace(indexes_list, spacing, physical_size, T_data2world)

    physical_points = np.reshape(physical_points, grid_size.append(3))

    return physical_points


def get_closest_points(pc1, pc2):
    """
    returns the points of pc1 which are closest to pc2
    """
    # kdtree=cKDTree(pc1[:,:3])
    # dist, ind =kdtree.query(pc2[:,:3], 1)
    # ind = ind.flatten()
    # points = pc1[ind, ...]

    kdtree=cKDTree(pc1[:,:3])
    dist, ind =kdtree.query(pc2[:,:3], 1)
    ind = ind.flatten()
    points = pc1[ind, ...]

    return ind, points


def get_grid_indexes(grid_size):
    a = [ list(range(0, size)) for size in grid_size]

    # getting the indexes list with the first dimension changing slowelier and last one changing faster
    indexes_list = list(itertools.product(*a))
    return np.array(indexes_list)


def volume2slices(volume, T_vol2world, vol_spacing, image_size, img_spacing, T_img2world):

    vol_colored = np.zeros(volume.shape)

    if not isinstance(T_img2world, list):
        T_img2world = [T_img2world]

    if len(image_size) == 2:
        image_size.append(1)

    vol_size = volume.shape
    vol_physical_size = np.array([vol_size[i] * vol_spacing[i] for i in range(3)])
    vol_indexes = get_grid_indexes(vol_size)
    vol_points = voxels2pyhsicalspace(vol_indexes, vol_spacing, vol_physical_size, T_vol2world)

    images_list = []
    for i, T_img in enumerate(T_img2world):

        t1 = time.time()
        img_indexes = get_grid_indexes(image_size)
        img_physical_size = np.array([image_size[i] * img_spacing[i] for i in range(3)])
        img_points = voxels2pyhsicalspace(img_indexes, img_spacing, img_physical_size, T_img)
        print("\nTime to find image physical indexes: ", time.time() - t1, " s")

        bounding_box = get_bounding_box(img_points)
        print(bounding_box)

        t1 = time.time()
        vol_idxes_in_box = find_indexes_in_box(vol_points, bounding_box)

        print("\nTime to find indexes in box: ", time.time() - t1, " s")

        image = np.zeros(image_size)

        if vol_idxes_in_box.size == 0:
            images_list.append(image)
            continue

        print("points in box: ", vol_idxes_in_box.size)
        vol_points_bb = vol_points[vol_idxes_in_box, ...]
        vol_indexes_bb = vol_indexes[vol_idxes_in_box, ...]

        t1 = time.time()
        ind, _ = get_closest_points(vol_points_bb, img_points)
        print("Time to find closest points: ", time.time() - t1, " s")

        vol_ind = vol_indexes_bb[ind, ...]

        image[img_indexes[:, 0], img_indexes[:, 1], img_indexes[:, 2]] = volume[
            vol_ind[:, 0], vol_ind[:, 1], vol_ind[:, 2]]

        vol_colored[vol_indexes_bb[:, 0], vol_indexes_bb[:, 1], vol_indexes_bb[:, 2]] = i

        images_list.append(image)

    return images_list, vol_colored


def main():
    imfusion.init()
    vol_path = "C:/Users/maria/OneDrive/Desktop/JaneSimulatedData/labelmap.imf"
    volume, = imfusion.open(vol_path)

    vol_spacing = volume[0].spacing
    vol_array = np.squeeze(np.array(volume))

    vol_array = np.transpose(vol_array, [2, 1, 0])
    vol_size = [item for item in vol_array.shape]
    T_vol2world = np.linalg.inv(volume[0].matrix)

    us_path = "C:/Users/maria/OneDrive/Desktop/JaneSimulatedData/Ultrasound11.imf"
    us_sweep, = imfusion.open(us_path)

    us_0 = us_sweep[0]
    us_spacing = us_0.spacing
    #us_size = [item for item in np.squeeze(np.array(us_0)).shape]

    us_size = [us_0.width, us_0.height]

    # us_sweep = np.squeeze(np.array(us_sweep))

    T_img2world_list = [us_sweep.matrix(i) for i in range(len(us_sweep))]
    images, vol = volume2slices(vol_array, T_vol2world, vol_spacing, us_size, us_spacing, T_img2world_list)
    print("finished processing")

    label_maps = imfusion.SharedImageSet()

    for image in images:

        image = np.transpose(image, [1, 0, 2]).astype(np.uint8)
        plt.imshow(image)
        plt.show()

        imfusion_image = imfusion.SharedImage(image)
        imfusion_image.spacing = us_spacing
        label_maps.add(imfusion_image)

    #label_maps.modality = imfusion.Data.Modality.LABEL
    imfusion.executeAlgorithm('IO;ImFusionFile', [label_maps],
                              imfusion.Properties({'location': "C:/Users/maria/OneDrive/Desktop/JaneSimulatedData/image_labelmap.imf"}))

    volume.setDirtyMem()
    vol_array_dirt = np.array(volume[0], copy=False)

    vol = np.transpose(vol, [2, 1, 0])
    vol_array_dirt[:, :, :, 0] = vol[:, :, :]

    imfusion.executeAlgorithm('IO;ImFusionFile', [volume],
                              imfusion.Properties({'location': "C:/Users/maria/OneDrive/Desktop/JaneSimulatedData/test_vol.imf"}))


main()
# size = [10, 5, 20]
# T_data2world = np.eye(4)
# T_data2world[0:3, -1] = [20, 0, 5]
# spacing = [1, 1, 1]
#
# volume = np.zeros([10, 10, 10])
# image_size = [5, 5]
# vol_spacing = [1, 1, 1]
# image_spacing = [1, 1, 1]
# T_vol2world = np.copy(T_data2world)
# T_img2world = T_data2world
#
# volume2slice(volume, vol_spacing, T_vol2world, image_size, image_spacing, T_img2world)







