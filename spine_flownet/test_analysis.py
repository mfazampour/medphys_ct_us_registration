import os
import numpy as np
import visualization_utils as vutils
from test_utils import umeyama_absolute_orientation, rigid_transform_3D


def get_color_code(color_name):
    color_code_dict = {
        "black": [0, 0, 0],
        "red": [170, 0, 0],
        "dark_green": [0, 85, 0],
        "blue": [0, 0, 127],
        "yellow": [211, 230, 38],
        "default": "1 1 0 1"
    }

    if color_name in color_code_dict.keys():
        color_norm = [str(item/255) for item in color_code_dict[color_name]]
        color_str = " ".join(color_norm) + " 1"
        return color_str

    else:
        return color_code_dict["default"]


def add_spine_vertbyvert(imf_root, point_cloud, color_array, name, save_path, data_uid):

    vert_dict = {}
    for i in range(1, 5):
        vert_dict["vert" + str(i)] = point_cloud[color_array==i]

    vertebrae_colors = ["red", "black", "yellow", "blue", "dark_green"]

    vert_keys = [item for item in vert_dict.keys()]
    for i, vert in enumerate(vert_keys):

        vert_color = get_color_code(vertebrae_colors[i])
        filepath = os.path.join(save_path, name + "_vert" + str(i) + ".txt")
        np.savetxt(filepath, vert_dict[vert])
        imf_root = vutils.add_block_to_xml(imf_root,
                                           parent_block_name="Annotations",
                                           block_name="point_cloud_annotation",
                                           param_dict={"referenceDataUid": "data" + str(data_uid),
                                                       "name": str(name) + "_" + vert,
                                                       "color": vert_color,
                                                       "labelText":"some",
                                                       "pointSize": "4"})

        imf_root =vutils.add_block_to_xml(imf_root,
                                          parent_block_name="Algorithms",
                                          block_name="load_point_cloud",
                                          param_dict={"location": os.path.split(filepath)[-1],
                                                      "outputUids": "data" + str(data_uid)})

        data_uid += 1

    return imf_root, data_uid


def save_for_pc_transformation(data_dir, file_id, save_root):
    """
    Saving the generated data in imfusion workspaces at specific location
    """

    source_pc = np.loadtxt( os.path.join(data_dir, "source_spine" + file_id + ".txt"))
    predicted_pc = np.loadtxt(os.path.join(data_dir, "predicted_spine" + file_id + ".txt"))
    gt_pc = np.loadtxt(os.path.join(data_dir, "gt_spine" + file_id + ".txt"))
    target_pc = np.loadtxt(os.path.join(data_dir, "target_spine" + file_id + ".txt"))

    data_uid = 0
    imf_tree, imf_root = vutils.get_empty_imfusion_ws()

    if not os.path.exists(os.path.join(save_root, file_id)):
        os.makedirs(os.path.join(save_root, file_id))

    imf_root, data_uid = add_spine_vertbyvert(imf_root=imf_root,
                                              point_cloud=target_pc[:, 0:3],
                                              color_array=target_pc[:, -1],
                                              name="target_pc",
                                              save_path=os.path.join(save_root, file_id),
                                              data_uid=data_uid)

    data_uid += 1

    for pc, name in zip([source_pc, predicted_pc, gt_pc], ["source_pc", "predicted_pc", "gt_pc"]):

        imf_root, data_uid = add_spine_vertbyvert(imf_root=imf_root,
                                                  point_cloud=pc[:, 0:3],
                                                  color_array=pc[:, -1],
                                                  name=name,
                                                  save_path=os.path.join(save_root, file_id),
                                                  data_uid=data_uid)

    vutils.write_on_file(imf_tree, os.path.join(save_root, file_id, "imf_ws.iws"))
    print("saved")


def get_points_by_color(pc, color):
    idxes = np.argwhere(pc[:, 3] == color).flatten()
    return pc[idxes, ...]


def get_vert_by_vert_transformation(source_pc, target_pc):

    transformation_dict = {}
    for vertebrae_level in range(1, 6):
        # source_vertebra = get_points_by_color(source_pc, vertebrae_level)
        # target_vertebra = get_points_by_color(target_pc, vertebrae_level)

        source_vertebra = np.transpose(get_points_by_color(source_pc, vertebrae_level))
        target_vertebra = np.transpose(get_points_by_color(target_pc, vertebrae_level))

        T = np.eye(4)
        # T[0:3, 0:3], T[0:3, -1] = umeyama_absolute_orientation(source_vertebra[:, 0:3], target_vertebra[:, 0:3])
        R, t = rigid_transform_3D(source_vertebra[0:3, :], target_vertebra[0:3, :])

        T[0:3, 0:3], T[0:3, -1] = R, np.squeeze(t)
        transformation_dict["vert_" + str(vertebrae_level)] = T

    return transformation_dict


def make_homogeneous(pc):
    if pc.shape[0] != 3:
        pc = np.transpose(pc)

    assert pc.shape[0] == 3

    return np.concatenate((pc, np.ones((1, pc.shape[1])) ), axis = 0 )


def apply_transform_vert_by_vert(pc, transformation_dict):
    transformed_pc = np.zeros(pc.shape)

    for i in range(1, 6):
        color_idxes = np.argwhere(pc == i).flatten()
        homogenous_points = make_homogeneous(pc[color_idxes, 0:3])
        transformed_points = np.matmul(transformation_dict["vert_" + str(i)], homogenous_points)[0:3, :]

        transformed_pc[color_idxes, 0:3] = np.transpose(transformed_points)
        transformed_pc[color_idxes, -1] = i

    return transformed_pc


def save_rigid_transformation(data_dir, file_id, save_root):
    """
    Saving the generated data in imfusion workspaces at specific location
    """

    source_pc = np.loadtxt( os.path.join(data_dir, "source_spine" + file_id + ".txt"))
    predicted_pc = np.loadtxt(os.path.join(data_dir, "predicted_spine" + file_id + ".txt"))
    gt_pc = np.loadtxt(os.path.join(data_dir, "gt_spine" + file_id + ".txt"))
    target_pc = np.loadtxt(os.path.join(data_dir, "target_spine" + file_id + ".txt"))

    data_uid = 0
    imf_tree, imf_root = vutils.get_empty_imfusion_ws()

    if not os.path.exists(os.path.join(save_root, file_id)):
        os.makedirs(os.path.join(save_root, file_id))

    imf_root, data_uid = add_spine_vertbyvert(imf_root=imf_root,
                                              point_cloud=target_pc[:, 0:3],
                                              color_array=target_pc[:, -1],
                                              name="target_pc",
                                              save_path=os.path.join(save_root, file_id),
                                              data_uid=data_uid)

    transformations = {
        "predicted_pc":get_vert_by_vert_transformation(source_pc, predicted_pc),
        "gt_pc": get_vert_by_vert_transformation(source_pc, gt_pc)
    }

    np.set_printoptions(precision=3, suppress=True)
    for i in range(1, 6):
        print(transformations["predicted_pc"]["vert_" + str(i)][0:3, 3], " --- ", transformations["gt_pc"]["vert_" + str(i)][0:3, 3])

    data_uid += 1
    for pc, name in zip([source_pc, predicted_pc, gt_pc], ["source_pc", "predicted_pc", "gt_pc"]):

        # Adding the point cloud
        imf_root, data_uid = add_spine_vertbyvert(imf_root=imf_root,
                                                  point_cloud=pc[:, 0:3],
                                                  color_array=pc[:, -1],
                                                  name=name,
                                                  save_path=os.path.join(save_root, file_id),
                                                  data_uid=data_uid)

        if name == "source_pc":
            continue

        # Adding the rigidly transformed vertebrae
        data_uid += 1
        transformed_pc = apply_transform_vert_by_vert(source_pc, transformations[name])
        imf_root, data_uid = add_spine_vertbyvert(imf_root=imf_root,
                                                  point_cloud=transformed_pc[:, 0:3],
                                                  color_array=transformed_pc[:, -1],
                                                  name=name + "_rigid",
                                                  save_path=os.path.join(save_root, file_id),
                                                  data_uid=data_uid)

    vutils.write_on_file(imf_tree, os.path.join(save_root, file_id, "imf_ws_rigid.iws"))
    print("saved")


def get_file_id(filename):
    return filename.split("spine")[-1].split(".")[0]


if __name__ == "__main__":

    root = "C:\\Repo\\thesis\\output\\flownet3d\\test_result"
    save_root = "temp_sanity_check"

    for file in os.listdir(root):

        save_for_pc_transformation(data_dir=root,
                                   file_id=get_file_id(file),
                                   save_root=save_root)

        save_rigid_transformation(data_dir=root,
                                  file_id=get_file_id(file),
                                  save_root=save_root)