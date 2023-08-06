import argparse
import json
import re

import numpy as np
import open3d as o3d
import pyvista as pv
from scipy.spatial import KDTree
import torch
import torch.nn.functional as F

import visualization_utils as utils

list_files = "/Users/janelameski/Desktop/jane/sofa/SOFAZIPPED/install/bin/" + "txtFiles/"


import os


# def extract_spine_id(filename):
#     """
#     Given a file, it extracts the id of the spine.
#
#     Example 1.
#
#     .. code-block:: console
#     >> filename = <spine_folder>\sspine1_vert1_1_0.txt
#     >> extract_spine_id(filename)
#     spine_1
#
#     Example 2:
#     >> filename = spine1_vert1_0.txt
#     >> extract_spine_id(filename)
#     spine_1
#
#     """
#
#     filename = os.path.split(filename)[-1]
#
#     return filename.split("_")[0]

# def extract_vertebra_id(filename):
#     """
#     Given a file, it extracts the id of the vertebra
#
#     Example 1.
#
#     .. code-block:: console
#     >> filename = <spine_folder>\sspine1_vert1_1_0.txt
#     >> extract_vertebra_id(filename)
#     vert1
#
#     Example 2:
#     >> filename = spine1_vert1_0.txt
#     >> extract_vertebra_id(filename)
#     vert1
#
#     """
#
#     filename = os.path.split(filename)[-1]
#     return filename.split("_")[1][0:5]


# def extract_timestamp_id(filename):
#     """
#     Given a file, it extracts the id of the timestamp
#
#     Example 1.
#
#     .. code-block:: console
#     >> filename = <spine_folder>\sspine1_vert1_1_0.txt
#     >> extract_timestamp_id(filename)
#     1_0
#
#     Example 2:
#     >> filename = spine1_vert1_1_0.txt
#     >> extract_timestamp_id(filename)
#     1_0
#
#     """
#     filename = os.path.split(filename)[-1]
#     spine_id = extract_spine_id(filename)
#     vertebra_id = extract_vertebra_id(filename)
#
#     timestamp_id = filename.replace(spine_id + "_" + vertebra_id, "")
#     timestamp_id = timestamp_id.split(".")[0]
#
#     return timestamp_id


class Point:
    def __init__(self, x, y, z, color):
        """
        :param: x: float: x coordinate of the point (in mm)
        :param: y: float: y coordinate of the point (in mm)
        :param: z: float: z coordinate of the point (in mm)
        :param: color: int: integer indicating the color of the point
        """
        self.x = x
        self.y = y
        self.z = z
        self.color = color

    @staticmethod
    def from_array(a):
        if a.shape[0] == 4:
            return Point(a[0], a[1], a[2], a[3])
        else:
            return Point(a[0], a[1], a[2], 0)

    def __str__(self):
        return f"[{self.x}, {self.y}, {self.z}, {self.color}]"

    def _get_pt_as_array(self):
        return np.array([self.x, self.y, self.z])

    def get_closest_point_in_cloud(self, pc, filter_by_color=True):

        distances = np.array(
            [np.linalg.norm(x + y + z) for (x, y, z) in np.abs(pc[:, :3] - self._get_pt_as_array())])

        if not filter_by_color:
            idx = distances.argmin()
            return idx, pc[idx]

        if len(np.where(pc[:, 3] == self.color)) == 0:
            return None, None

        distances[pc[:, 3] != self.color] = np.max(distances) + 1

        idx = distances.argmin()

        return idx, pc[idx]


def indexes2points(idxes_list, point_cloud, color=0):
    """
    Converts a list of indexes or a index to a set of Points, extracting the point coordinates from the input
    point_cloud

    :param: idxes_list: list(int): list of input indexes
    :param: point_cloud: np.ndarray of size [Nx3] or [Nx4]. If the array size is [Nx4], the last dimension is considered
        to be the color of the point
    :param: color: if the array has size [Nx3], the the color of all the points in the point cloud is set to color
        (Default to 0).
    :return: a list of Point objects, containing the 3d coordinates and color of the input point cloud at the input
        indexes

    Example:
        .. code-block:: console
            >> idxes_list = [1, 3]
            >> point_cloud = np.ndarray([ 10, 14, 20, 1
                                        [ 10, 30, 20, 4],
                                        [ 18, 20, 18, 2],
                                        [40, 1, 20, 2])

            >> indexes2points(idxes_list, point_cloud)
            [Point(x = 10, y = 30, z = 4, color = 1), Point(x = 40, y=1, z = 20, color = 2)]
    """

    if point_cloud.shape[1] > 3:
        color = point_cloud[:, 3]
    else:
        color = np.ones([point_cloud.shape[0], ]) * color

    if isinstance(idxes_list, int) or isinstance(idxes_list, float):
        idxes_list = [idxes_list]

    constraints_points = []
    for item in idxes_list:
        if isinstance(item, tuple) or isinstance(item, list):
            assert all(isinstance(x, int) for x in item) or all(isinstance(x, float) for x in item)

            constraints_points.append(tuple(Point(x=point_cloud[idx, 0],
                                                  y=point_cloud[idx, 1],
                                                  z=point_cloud[idx, 2],
                                                  color=color[idx]) for idx in item))

        else:
            constraints_points.append(Point(x=point_cloud[item, 0],
                                            y=point_cloud[item, 1],
                                            z=point_cloud[item, 2],
                                            color=color[item]))

    if len(constraints_points) == 1:
        return constraints_points[0]

    return constraints_points


def points2indexes(point_list, point_cloud):
    """
    Converts a list of indexes or a points to a set of indexes, corresponding to indexes of the closest points in the
    source point cloud.

    :param: point_list: list(Point): list of input Point
    :param: point_cloud: np.ndarray of size [Nx3]. If the number of columns is higher than 3 (e.g. the input array
        has size [Nx4], then only the first 3 columns are considered)

    Example:
        .. code-block:: console
            >> idxes_list = [Point(x = 11, y = 29, z = 3, color = 1), Point(x = 41, y=1, z = 20, color = 2)]
            >> point_cloud = np.ndarray([ 10, 14, 20, 1
                                        [ 10, 30, 20, 4],
                                        [ 18, 20, 18, 2],
                                        [40, 1, 20, 2])

            >> indexes2points(idxes_list, point_cloud)
            [1, 3]
    """

    idxes_list = []

    for item in point_list:
        if isinstance(item, tuple) or isinstance(item, list):
            assert all(isinstance(x, Point) for x in item)
            idxes_list.append(tuple(p.get_closest_point_in_cloud(point_cloud)[0] for p in item))

        else:
            idxes_list.append(item.get_closest_point_in_cloud(point_cloud[0]))

    return idxes_list


def points2indexes_exact(point_list, point_cloud, is_point=True):
    """
    Converts a list of indexes or a points to a set of indexes, corresponding to indexes of the closest points in the
    source point cloud.

    :param: point_list: list(Point): list of input Point
    :param: point_cloud: np.ndarray of size [Nx3]. If the number of columns is higher than 3 (e.g. the input array
        has size [Nx4], then only the first 3 columns are considered)

    Example:
        .. code-block:: console
            >> idxes_list = [Point(x = 11, y = 29, z = 3, color = 1), Point(x = 41, y=1, z = 20, color = 2)]
            >> point_cloud = np.ndarray([ 10, 14, 20, 1
                                        [ 10, 30, 20, 4],
                                        [ 18, 20, 18, 2],
                                        [40, 1, 20, 2])

            >> indexes2points(idxes_list, point_cloud)
            [1, 3]
    """

    # if is_point:
    p1_list, p2_list = zip(*point_list)
    p1_list = [p._get_pt_as_array() for p in p1_list]
    p2_list = [p._get_pt_as_array() for p in p2_list]
    kdtree = KDTree(point_cloud[:, :3])
    _, idx_1 = kdtree.query(p1_list, 1)
    _, idx_2 = kdtree.query(p2_list, 1)
    idxes_list = list(zip(idx_1, idx_2))

    return idxes_list


def obtain_indices_raycasted_original_pc(spine_target, r_target):
    """
    Find indices in spine_target w.r.t. r_target such that they are the closest points between the two
    point clouds

    :param: spine_target: np.ndarray with size [Nx3] with the point cloud for which the closest point indexes are
        extracted If the second dimension is higher then 3, only the first 3 dimensions are considered
    :param: r_target: np.ndarray with size [Nx3] with the point cloud used to find the closest points in spine_target.
        If the second dimension is higher then 3, only the first 3 dimensions are considered

    Example:

        .. code-block:: console
            >> spine_target = np.ndarray([ 10, 14, 20, 1
                                [ 10, 30, 20, 4],
                                [ 18, 20, 18, 2],
                                [40, 1, 20, 2])

            >> r_target = np.ndarray([ 18, 21, 18, 2],
                                     [40, 1, 20, 3])

            >> obtain_indices_raycasted_original_pc(spine_target, r_target)
            [2, 3]
    """
    kdtree = KDTree(spine_target[:, :3])
    dist, points = kdtree.query(r_target[:, :3], 1)

    return list(set(points))


def create_source_target_with_vertebra_label(source_pc, target_pc, vert):
    """
    source_pc: source point cloud
    target_pc: target point cloud
    vert: [1-5] for [L1-L5] vertebra respectively

    this function is to create source and target point clouds with label for each vertebra
    """
    source_pc = np.array(source_pc.points)
    target_pc = np.array(target_pc.points)
    source = np.ones((source_pc.shape[0], source_pc.shape[1] + 1))
    source[:, :3] = source_pc
    source[:, 3] = source[:, 3] * vert
    target = np.ones((target_pc.shape[0], target_pc.shape[1] + 1))
    target[:, :3] = target_pc
    target[:, 3] = target[:, 3] * vert

    return source, target


def get_color_code(color_name):
    color_code_dict = {
        "dark_green": "0 0.333 0 1",
        "yellow": "1 1 0 1",
        "default": "1 1 0 1"
    }

    if color_name in color_code_dict.keys():
        return color_code_dict[color_name]

    else:
        return color_code_dict["default"]


def save_for_sanity_check(data, save_dir):
    """
    Saving the generated data in imfusion workspaces at specific location
    """

    source_pc = data["source_pc"][:, :3]
    target_pc = data["target_pc"][:, :3]

    gt_target_pc = source_pc + data["flow"]

    save_folder_path = os.path.join(save_dir, data["spine_id"], data["target_ts_id"])
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)

    # saving the point clouds
    # 1. Saving the full point clouds
    np.savetxt(os.path.join(save_folder_path, "full_source_pc.txt"), source_pc[:, :3])
    np.savetxt(os.path.join(save_folder_path, "full_target_pc.txt"), target_pc[:, :3])
    np.savetxt(os.path.join(save_folder_path, "full_gt_pc.txt"), gt_target_pc[:, :3])

    ps_list = [("full_source_pc", os.path.join(save_folder_path, "full_source_pc.txt"),
                get_color_code("dark_green")),
               ("full_target_pc", os.path.join(save_folder_path, "full_target_pc.txt"),
                get_color_code("yellow")),
               ("full_gt_pc", os.path.join(save_folder_path, "full_gt_pc.txt"),
                get_color_code("yellow"))]

    imf_tree, imf_root = utils.get_empty_imfusion_ws()

    for i, (name, path, color) in enumerate(ps_list):
        imf_root = utils.add_block_to_xml(imf_root,
                                          parent_block_name="Annotations",
                                          block_name="point_cloud_annotation",
                                          param_dict={"referenceDataUid": "data" + str(i),
                                                      "name": str(name),
                                                      "color": str(color),
                                                      "labelText": "some",
                                                      "pointSize": "2"})

        imf_root = utils.add_block_to_xml(imf_root,
                                          parent_block_name="Algorithms",
                                          block_name="load_point_cloud",
                                          param_dict={"location": path,
                                                      "outputUids": "data" + str(i)})

    # Adding the biomechanical_constraints

    for i, (c1, c2) in enumerate(data["biomechanical_constraint"]):
        c1_idx, _ = c1.get_closest_point_in_cloud(data["source_pc"], filter_by_color=True)
        c2_idx, _ = c2.get_closest_point_in_cloud(data["source_pc"], filter_by_color=True)

        p1 = data["source_pc"][c1_idx, :3]
        p2 = data["source_pc"][c2_idx, :3]
        points = " ".join([str(item) for item in p1]) + " " + " ".join([str(item) for item in p2])
        imf_root = utils.add_block_to_xml(imf_root,
                                          parent_block_name="Annotations",
                                          block_name="segment_annotation",
                                          param_dict={"name": "constraint_" + str(i + 1),
                                                      "points": points})

    utils.write_on_file(imf_tree, os.path.join(save_folder_path, "imf_ws.iws"))


def read_vert_pcs(subfolder_path_vert, raycasted=False, source=False, nr_deform=-1, use_net_output=False):
    lumbar_vertebrae = ["verLev20", "verLev21", "verLev22", "verLev23", "verLev24"]

    vert_pcs = {}
    for level in lumbar_vertebrae:
        obj_folder = f"{subfolder_path_vert}_{level}"
        # Find the non-deformed mesh
        if not raycasted:
            pcd_files = [
                file_name
                for file_name in os.listdir(obj_folder)
                if file_name.endswith(
                    '.obj') and 'centered' not in file_name and 'scaled' not in file_name and 'deformed' not in file_name
            ]
        elif raycasted and not source:
            if use_net_output:
                pcd_files = [
                    file_name
                    for file_name in os.listdir(obj_folder)
                    if file_name.endswith(
                        '.pcd') and f"forces{nr_deform}" in file_name and "deformed" in file_name and "net_output" in file_name
                ]
            else:
                pcd_files = [
                    file_name
                    for file_name in os.listdir(obj_folder)
                    if file_name.endswith(
                        '.pcd') and f"forces{nr_deform}" in file_name and "deformed" in file_name and "raycast" in file_name
                ]
        else:
            pcd_files = [
                file_name
                for file_name in os.listdir(obj_folder)
                if file_name.endswith(
                    '.pcd') and f"forces{nr_deform}" in file_name and "deformed" not in file_name and "raycast" in file_name
            ]
        if len(pcd_files) != 1:
            print(f"found {len(pcd_files)} meshes for vertebra {level} in {subfolder_path_vert} for deformation {nr_deform}")
            return None

        number = re.findall(r'\d+', level)[0]

        if raycasted:
            vert_pcs[f"vert{int(number)-19}"] = o3d.io.read_point_cloud(os.path.join(obj_folder, pcd_files[0]))
        else:
            vert_pcs[f"vert{int(number)-19}"] = pv.read(os.path.join(obj_folder, pcd_files[0]))

    return vert_pcs


def read_json_file(root_path_vert, line, source_vertebrae):
    spring_folder = os.path.join(root_path_vert, "spring_files/")

    spring_file = os.path.join(spring_folder, f"sub-{line}.json")

    # Read the JSON file
    with open(spring_file, 'r') as file:
        data = json.load(file)

    # Initialize an empty dictionary
    result = []
    facet = {}

    levels = {'vert1':1, 'vert2':2, 'vert3':3, 'vert4':4, 'vert5':5}
    # Iterate over each key in the "springs" dictionary
    for key, value in data['springs'].items():
        points = []

        if 'v0' in key or 'v6' in key:
            continue

        key1 = [key_ for key_ in source_vertebrae.keys() if key[1] in key_][0]
        key2 = [key_ for key_ in source_vertebrae.keys() if key[3] in key_][0]

        print(f'key1-key2 is {key1}-{key2}')

        vert1_points = source_vertebrae[key1].points.__array__()
        vert1_points = np.concatenate((vert1_points, np.ones((vert1_points.shape[0], 1)) * levels[key1]), axis=1)
        vert2_points = source_vertebrae[key2].points.__array__()
        vert2_points = np.concatenate((vert2_points, np.ones((vert2_points.shape[0], 1)) * levels[key2]), axis=1)

        # Extract the start and end points for each key
        for _, sub_value in value.items():
            v_data = sub_value.split()
            points += [(Point.from_array(vert1_points[int(v_data[i]), :]), Point.from_array(vert2_points[int(v_data[i + 1]), :])) for i in range(0, len(v_data), 5)]

        if key1 not in facet.keys():
            facet[key1] = []
        if key2 not in facet.keys():
            facet[key2] = []
        facet[key1].append(Point.from_array(vert1_points[int(value['facet_left'].split()[0]), :]))
        facet[key1].append(Point.from_array(vert1_points[int(value['facet_right'].split()[0]), :]))

        facet[key2].append(Point.from_array(vert1_points[int(value['facet_left'].split()[1]), :]))
        facet[key2].append(Point.from_array(vert1_points[int(value['facet_right'].split()[1]), :]))
        # Store the points in the result dictionary with the corresponding key
        result += points

    return result, facet


def read_full_spine_mesh(subfolder, nr_deform=-1):
    if nr_deform >= 0:
        pc = [
            file_name
            for file_name in os.listdir(subfolder)
            if file_name.endswith('.obj') and 'deformed' in file_name and f"field{nr_deform}" in file_name and "centered" not in file_name
        ]
    else:
        pc = [
            file_name
            for file_name in os.listdir(subfolder)
            if file_name.endswith('.obj') and 'deformed' not in file_name and "centered" not in file_name and "_lumbar_" in file_name
        ]

    if len(pc) != 1:
        raise Exception(f"no pc found in {subfolder} for field {nr_deform}, {len(pc)}")

    pc = pv.read(os.path.join(subfolder, pc[0]))
    return pc


def match_the_flow(flow, preprocessed_source_spine, full_source_w_level):
    kdtree = KDTree(full_source_w_level[:, :3])
    _, points = kdtree.query(preprocessed_source_spine[:, :3], 1)

    sub_flow = flow[points, :]
    return sub_flow


def preprocess_spine_data_new(root_path_spine, root_path_vert, line, number_of_deformations, use_net_output: bool):
    """
    Preprocess the data for a given spine dataset. Specifically, for the given spine (i.e. for a given spine_id),
    it ierates over all the timestamps for that given spine.
    The function does the following.
    1. It loads the "ts0" as the timestamp of the underformed spine, and therefore of the source spine.
    2. It loads the biomechanical constraints for the given spine
    3. It iterates over all the timestamps different from t0, where the spine is considered to be deformed compared to
        t0, and for each timestamp different from ts0:
        3.a Computes the flow from the source to the target points, assuming a correspondence
            between points at different timestamps
        3.b Concatenates all the vertebrae together for both source (ts0) and target (considered timestamp),
            indicating the vertebral level in the resulting concatenated point clouds in a 4th column,
            where L1 is indicated with 1, L2 with 2, L3 with 3,
            L4 with 4, L5 with 5.
        3.c. For each given source-deformed spine pair, generate a Data dict with the following keys:
            "spine_id", "source_ts_id", "target_ts_id", "source_pc", "target_pc", "flow", "biomechanical_constraint"

    """

    subfolder_path_spine = os.path.join(root_path_spine, f"sub-{line}/")
    subfolder_path_vert = os.path.join(root_path_vert, f"sub-{line}")

    # find the files needed: vertebrae pc, spine pc, deformed spine pc, json file of the springs

    # read non deformed full spine
    full_source_spine = read_full_spine_mesh(subfolder_path_spine, nr_deform=-1)

    sorting = np.lexsort((full_source_spine.points[:, 1], full_source_spine.points[:, 2]))

    # read non deformed full vertebrae pcd
    source_vertebrae_full = read_vert_pcs(subfolder_path_vert, raycasted=False, source=True, nr_deform=-1)
    if source_vertebrae_full is None:
        return None
    full_source_w_level = []

    for i, vertebra in enumerate(["vert1", "vert2", "vert3", "vert4", "vert5"]):
        l_ = len(source_vertebrae_full[vertebra].points)
        full_source_w_level.append(np.concatenate((source_vertebrae_full[vertebra].points, np.ones((l_, 1)) * i + 1), axis=1))
    # full_source_w_level = np.concatenate(full_source_w_level, axis=0)[sorting]
    full_source_w_level = np.concatenate(full_source_w_level, axis=0)
    # Load the biomechanical constraints for the selected spine. biomechanical_constraints is loaded as a list
    # of tuples (Point, Point). biomechanical_constraints = [(Point, Point), (Point, Point), ..., (Point, Point)]
    # For a given tuple, the first element is the point from which the spring starts, the second point is the point
    # where the spring ends. Note that the biomechanical_constraints contain tuple defining the 3D position of the
    # constraints, and not their indexes.
    biomechanical_constraints, facets = read_json_file(root_path_vert, line, source_vertebrae_full)

    data = []
    for num in range(number_of_deformations):
        # Getting the source vertebrae dict, as {"vert1" : np.array(..), "vert2" : np.array(..),
        # "vert3" : np.array(..), "vert4" : np.array(..), "vert5" : np.array(..)}
        source_vertebrae = read_vert_pcs(subfolder_path_vert, raycasted=True, source=True, nr_deform=num)
        if source_vertebrae is None:
            continue

        # Getting the target vertebrae dict, as {"vert1" : np.array(..), "vert2" : np.array(..),
        # "vert3" : np.array(..), "vert4" : np.array(..), "vert5" : np.array(..)}
        deformed_vertebrae = read_vert_pcs(subfolder_path_vert, raycasted=True, source=False, nr_deform=num, use_net_output=use_net_output)
        if deformed_vertebrae is None:
            continue

        # Preprocess the point clouds of each given vertebra and then concatenate the vertebrae in a single point cloud
        preprocessed_source_vertebrae = []
        preprocessed_target_vertebrae = []
        for i, vertebra in enumerate(["vert1", "vert2", "vert3", "vert4", "vert5"]):
            preprocessed_source_pc, preprocessed_target_pc = \
                create_source_target_with_vertebra_label(source_pc=source_vertebrae[vertebra],
                                                         target_pc=deformed_vertebrae[vertebra],
                                                         vert=i + 1)
            preprocessed_source_vertebrae.append(preprocessed_source_pc)
            preprocessed_target_vertebrae.append(preprocessed_target_pc)

        # Concatenating source and target vertebrae into a single spine point cloud
        preprocessed_source_spine = np.concatenate(preprocessed_source_vertebrae)
        threshold = int(2e4)
        if preprocessed_source_spine.shape[0] > threshold:
            preprocessed_source_spine = preprocessed_source_spine[np.random.choice(preprocessed_source_spine.shape[0], threshold)]
        preprocessed_target_spine = np.concatenate(preprocessed_target_vertebrae)
        if preprocessed_target_spine.shape[0] > threshold:
            preprocessed_target_spine = preprocessed_target_spine[np.random.choice(preprocessed_target_spine.shape[0], threshold)]

        deformed_spine = read_full_spine_mesh(subfolder_path_spine, nr_deform=num)

        # full_flow = deformed_spine.points[sorting] - full_source_spine.points[sorting]
        full_flow = deformed_spine.points - full_source_spine.points

        print(f"for spine {line} and deformation {num}: avg. def: {np.mean(full_flow)} and max. def: {np.max(np.abs(full_flow))}")
        # find the flow for the nodes in preprocessed_source_spine
        flow = match_the_flow(full_flow, preprocessed_source_spine, full_source_w_level)


        # Append the generated source-target pair to the data list
        data_ = {
            "spine_id": line,
            "source_ts_id": "source",  # todo: remove this one from data
            "target_ts_id": "field" + str(num),  # todo: change to show num
            "source_pc": preprocessed_source_spine,
            "target_pc": preprocessed_target_spine,
            "flow": flow,
            "full_flow": full_flow,
            "biomechanical_constraint": biomechanical_constraints,
            "full_source_pc": full_source_spine.points, #[sorting],
            "full_deformed_pc": deformed_spine.points, #[sorting],
            "full_source_w_level": full_source_w_level
        }

        if not rigidity_check(data_["source_pc"], data_["flow"]):
            continue

        add_biomechanical_constraints_to_raycasted(data_)

        if not rigidity_check(data_["source_pc"], data_["flow"]):
            print("not rigid after constraints")
            continue

        data.append(data_)

    return data, facets


def rigidity_check(preprocessed_source_spine, flow):

    points_source = torch.tensor(preprocessed_source_spine[preprocessed_source_spine[:, 3] == 1])[None, ...]
    dist_source = torch.cdist(points_source, points_source).view(-1)
    flow_pred = torch.tensor(flow[preprocessed_source_spine[:, 3] == 1])[None, ...]
    points_pred = points_source[..., :3] + flow_pred
    dist_pred = torch.cdist(points_pred, points_pred).view(-1)
    rigidity = F.mse_loss(dist_pred, dist_source)
    print(f"rigidity loss is: {rigidity}")
    if rigidity > 3e-1:
        print("Flow is not rigid anymore")
        return False
    else:
        return True


def add_biomechanical_constraints_to_raycasted(data):
    # Getting the flow at the biomechanical_constraints points as it will be needed later
    constraint_indexes = points2indexes_exact(point_list=data["biomechanical_constraint"],
                                              point_cloud=data["full_source_w_level"], is_point=True)

    constraint_points, constraint_flows = [], []
    for i, (p1_idx, p2_idx) in enumerate(constraint_indexes):
        p1_colored, p2_colored = data["full_source_w_level"][p1_idx, :], data["full_source_w_level"][p2_idx, :]
        p1_flow, p2_flow = data["full_flow"][p1_idx, :], data["full_flow"][p2_idx, :]

        if p1_colored[3] + 1 != p2_colored[3]:
            continue

        if sum((data["biomechanical_constraint"][i][0]._get_pt_as_array() - p1_colored[:3]) ** 2) > 1e-5:
            print("shit")

        if (data["biomechanical_constraint"][i][0].color - p1_colored[3]) != 0:
            print(i, data["biomechanical_constraint"][i][0].color, p1_colored[3])
            # p1_colored[3] = data["biomechanical_constraint"][i][0].color
            print("shit1")
        if (data["biomechanical_constraint"][i][1].color - p2_colored[3]) != 0:
            print(i, data["biomechanical_constraint"][i][1].color, p2_colored[3])
            # p2_colored[3] = data["biomechanical_constraint"][i][1].color
            print("shit2")

        if sum((data["biomechanical_constraint"][i][1]._get_pt_as_array() - p2_colored[:3]) ** 2) > 1e-5:
            print("shit3")

        constraint_points.append((p1_colored, p2_colored))
        constraint_flows.append((p1_flow, p2_flow))

    # # Getting the indexes of the points in the source data which are closest to the ray_casted source points
    # source_ray_casted_idxes = obtain_indices_raycasted_original_pc(spine_target=data["full_source_pc"],
    #                                                                r_target=data["source_pc"])
    # # data["source_pc"] = data["source_pc"][source_ray_casted_idxes]
    # data["flow"] = data["flow"][source_ray_casted_idxes]

    # # Getting the indexes of the points in the target data which are closest to the ray_casted target points
    # target_ray_casted_idxes = obtain_indices_raycasted_original_pc(spine_target=data["full_deformed_pc"],
    #                                                                r_target=data["target_pc"])
    # data["target_pc"] = data["target_pc"][target_ray_casted_idxes]

    # Adding the biomechanical constraints to the source as they might be not present due to the ray-casting
    p1, p2 = zip(*constraint_points)
    # sanity check: this list should be empty:
    # [(p1_, p2_) for p1_, p2_ in zip(p1, p2) if p1_[3] + 1 != p2_[3]]

    data["source_pc"] = np.concatenate((data["source_pc"], np.reshape(p1, [p1.__len__(), 4])))
    data["source_pc"] = np.concatenate((data["source_pc"], np.reshape(p2, [p2.__len__(), 4])))

    flow1, flow2 = zip(*constraint_flows)
    data["flow"] = np.concatenate((data["flow"], np.reshape(flow1, [flow1.__len__(), 3])))
    data["flow"] = np.concatenate((data["flow"], np.reshape(flow2, [flow2.__len__(), 3])))

    return data


def save_facets(facets, save_path):
    if os.path.exists(save_path):
        os.remove(path=save_path)

    for idx, key in enumerate(facets.keys()):
        # Read the lines from the text file
        for point in facets[key]:
            with open(save_path, 'a') as file:
                file.write(f"{point.x} {point.y} {point.z} {idx + 1}\n")


def generate_npz_files(root_path_spine, root_path_vert, txt_file, dst_npz_path, number_of_deformations, use_net_output: bool):
    if not os.path.exists(dst_npz_path):
        os.makedirs(dst_npz_path)

    # Read the lines from the text file
    with open(txt_file, 'r') as file:
        lines = file.readlines()

    # Process each line
    for line in lines:
        line = line.strip()  # Remove leading/trailing whitespaces

        print(f"processing folder {line}")

        data_array, facets = preprocess_spine_data_new(root_path_spine, root_path_vert, line, number_of_deformations, use_net_output)

        if data_array is None:
            continue

        print(f"Read {data_array.__len__()} data in folder {line}")

        save_facets(facets, os.path.join(dst_npz_path, "facet_" + line + ".txt"))

        for data in data_array:
            # if ray_casted:  # todo: ray casted is not implemented yet
            #     data = get_ray_casted_data(data, src_raycasted_pc_path)

            # add_biomechanical_constraints_to_raycasted(data)

            # convert biomechanical_constraint to a 1-d array, putting all the constraint on a single
            # row - this needs to be changed in future to be a list of tuple or similar format where it is clear
            # which point belongs to the same connecting spring

            constraint_indexes = points2indexes_exact(data["biomechanical_constraint"], data["source_pc"], is_point=True)

            flattened_constraints = [i for sub in constraint_indexes for i in sub]
            np.savez_compressed(file=os.path.join(dst_npz_path,
                                                  "full_" + line + "_" + data["target_ts_id"] + ".npz"),
                                flow=data["flow"],
                                pc1=data["source_pc"],
                                pc2=data["target_pc"],
                                ctsPts=flattened_constraints)


if __name__ == "__main__":
    """
        # example setup
        root_vertebrae = "/home/miruna20/Documents/Thesis/sofa/vertebrae/train"
        spine_name = "sub-verse500"
        txt_file = "../samples/test.txt"
    """

    arg_parser = argparse.ArgumentParser(description="Extrapolate the deformation field to cover the whole image")

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
        "--root_path_vert",
        required=True,
        help="folder containing the data for vertebrae"
    )
    arg_parser.add_argument(
        "--nr_deform_per_spine",
        required=True,
        type=int,
        help="Number of deformations per spine"
    )

    arg_parser.add_argument(
        "--output_folder",
        required=True,
        help="path to the output folder"
    )

    arg_parser.add_argument(
        "--use_net_output",
        action='store_true',
        default=False,
        help="use the segmentation output of the network"
    )

    print("Preprocessing to get the network ready data")
    #
    args = arg_parser.parse_args()

    generate_npz_files(args.root_path_spine, args.root_path_vert, args.txt_file,
                       args.output_folder, args.nr_deform_per_spine, args.use_net_output)
