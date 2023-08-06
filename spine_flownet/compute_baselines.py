import time

from data import SceneflowDataset, VerseFlowDataset
import argparse
from constrained_cpd.BiomechanicalCPD import BiomechanicalCpd
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from test_utils.metrics import umeyama_absolute_orientation, pose_distance, np_chamfer_distance
import os
from sklearn.neighbors import KDTree
from sklearn.metrics import mean_squared_error
import wandb


def get_closest_points(pc1, pc2):
    """
    returns the points of pc1 which are closest to pc2
    """
    kdtree = KDTree(pc1[:, :3])
    dist, ind = kdtree.query(pc2[:, :3], 1)
    ind = ind.flatten()
    points = pc1[ind, ...]

    return points


def visualize(iteration, error, X, Y, ax):
    plt.cla()
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], color='red', alpha=0.1)
    ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], color='blue', alpha=0.1)
    ax.text2D(0.87, 0.92, 'Iteration: {:d}\nQ: {:06.4f}'.format(
        iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
              fontsize='x-large')
    # ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)


def get_connected_idxes(constraint):
    """
    From a list of connections as [idx_0, idx_1, idx_2, idx_3, ..., idx_m] it returns a list of tuples containing
    connecting indexes, assuming that the 2*i index in the list is connected with the 2*i+1 index.

    Example:
        constraint = [idx_0, idx_1, idx_2, idx_3, ..., idx_{2m}]
        returned value = [(idx_0, idx_1), (idx_2, idx_3), (idx_4, idx_5), ..., (idx_{2m-1}, idx_{2m})]
    """
    constrain_pairs = []
    for j in range(0, len(constraint) - 1, 2):
        constrain_pairs.append((constraint[j], constraint[j + 1]))

    return constrain_pairs


def order_connection(item, vertebral_level_idxes):
    """
    Given  the connection item = (connection_index_1, connection_index_2), indicating a connection between two points
    (indexes) in a point cloud, the function first detects which of the connection points node belongs to the input
    vertebra (i.e. which index is contained in vertebral_level_idxes indicating the indexes of the points in the cloud
    belonging to a given vertebra). If the first point in the tuple is the one belonging to the input vertebra, the
    function returns the input item with the same order. Otherwise, it returns the input item with swapped elements,
    in a way that the first element in the return item (connection) is always the point belonging to the input
    vertebra
    """
    if item[0] in vertebral_level_idxes:
        return item

    return item[1], item[0]


def get_springs_from_vertebra(vertebral_level_idxes, constraints_pairs):
    """
    It returns the list of connection starting from the input vertebral level as a list of tuples like:
    [(idx_current_vertebra_level_0, idx_connected_vertebra_level_0),
    (idx_current_vertebra_level_1, idx_connected_vertebra_level_2),
                                ...,
    (idx_current_vertebra_level_n, idx_connected_vertebra_level_n)]
    """
    current_vertebra_springs = [item for item in constraints_pairs if item[0] in vertebral_level_idxes
                                or item[1] in vertebral_level_idxes]

    current_vertebra_springs = [order_connection(item, vertebral_level_idxes) for item in current_vertebra_springs]
    return current_vertebra_springs


def read_batch_data(data):
    pc1, pc2, color1, color2, flow, mask1, constraint, position1, position2, file_name, tre = data
    pc1 = pc1.transpose(2, 1).contiguous().float()
    pc2 = pc2.transpose(2, 1).contiguous().float()
    color1 = color1.transpose(2, 1).contiguous().float()
    color2 = color2.transpose(2, 1).contiguous().float()
    flow = flow.transpose(2, 1).contiguous()
    # mask1 = mask1.cuda().float()
    # constraint = constraint.cuda()
    return color1, color2, constraint, flow, pc1, pc2, position1, file_name, tre


def get_gt_transform(source_pc, gt_flow):
    R_gt, t_gt = umeyama_absolute_orientation(from_points=source_pc,
                                              to_points=source_pc + gt_flow, fix_scaling=True)

    T = np.eye(4)
    T[0:3, 0:3] = R_gt
    T[0:3, -1] = t_gt
    return T


def get_fig_ax():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.set_xlim([-40, 40])
    ax.set_ylim([-40, 40])
    ax.set_zlim([0, 200])

    return fig, ax


def run_registration(cpd_method, with_callback=False):
    if with_callback:
        fig, ax = get_fig_ax()
        callback = partial(visualize, ax=ax)
        TY, (_, R_reg, t_reg) = cpd_method.register(callback)
        plt.close(fig)
    else:
        TY, (_, R_reg, t_reg) = cpd_method.register()

    T = np.eye(4)
    T[0:3, 0:3] = np.transpose(R_reg)
    T[0:3, -1] = t_reg

    return TY, T


def get_result_dict(source, gt_flow, predicted_pc, predicted_T, tre_points, position=None):
    result = []

    if position is None:
        position = [[i for i in range(source.shape[0])]]
        tre_points[:, -1] = 1

    for i, vertebral_level_idxes in enumerate(position):
        # 2.a Extracting the points belonging to the first vertebra
        current_vertebra = source[vertebral_level_idxes, ...]
        current_flow = gt_flow[vertebral_level_idxes, ...]
        predicted_vertebra = predicted_pc[vertebral_level_idxes, ...]
        gt_T = get_gt_transform(source_pc=current_vertebra,
                                gt_flow=current_flow)

        translation_distance, quaternion_distance = pose_distance(gt_T, predicted_T)
        mse_loss = mean_squared_error(current_vertebra + current_flow, predicted_vertebra)
        chamfer_dist = np_chamfer_distance(current_vertebra + current_flow, predicted_vertebra)

        # computing tre loss
        vertebra_target = tre_points[tre_points[:, -1] == i + 1]
        vertebra_target[:, -1] = 1  # making the points homogeneous
        vertebra_target = np.transpose(vertebra_target)
        gt_registered_target = np.matmul(gt_T, vertebra_target)  # Nx4
        predicted_registered_target = np.matmul(predicted_T, vertebra_target)  # Nx4
        tre = np.linalg.norm(gt_registered_target - predicted_registered_target, axis=0)

        res_ = {'mse loss': mse_loss,
                       'Chamfer Distance': chamfer_dist,
                       'translation distance': translation_distance,
                       'quaternion distance': quaternion_distance,
                       'TRE': np.mean(tre)}
        wandb.log(res_)
        print("tre: ", np.mean(tre))

        result.append(res_)

    return result


def preprocess_input(source_pc, gt_flow, position1, constrain_pairs, tre_points):
    vertebra_dict = []

    for i, vertebral_level_idxes in enumerate(position1):
        # 2.a Extracting the points belonging to the first vertebra
        current_vertebra = source_pc[vertebral_level_idxes, ...]
        current_flow = gt_flow[vertebral_level_idxes, ...]

        # 2.b Getting all the springs connections starting from the current vertebra
        current_vertebra_springs = get_springs_from_vertebra(vertebral_level_idxes, constrain_pairs)

        # 2.3 Generating the pairs: (current_vertebra_idx, constraint_position) where current_vertebra_idx
        # is the spring connection in the current_vertebra object and constraint_position is the position ([x, y, z]
        # position) of the point connected to the spring
        current_vertebra_connections = [(np.argwhere(vertebral_level_idxes == item[0]), source_pc[item[1]])
                                        for item in current_vertebra_springs]

        gt_T = get_gt_transform(source_pc=current_vertebra,
                                gt_flow=current_flow)

        tre_point = tre_points[tre_points[:, -1] == i + 1, :]

        vertebra_dict.append({'source': current_vertebra,
                              'gt_flow': current_flow,
                              'springs': current_vertebra_connections,
                              'gt_transform': gt_T,
                              'tre_points': tre_point})

    return vertebra_dict


def save_data(data_dict, save_path, postfix=""):
    if not os.path.exists(os.path.join(save_path, postfix)):
        os.makedirs(os.path.join(save_path, postfix))

    for key in data_dict.keys():
        np.savetxt(os.path.join(save_path, postfix, key + ".txt"), data_dict[key])


def make_homogeneous(pc):
    if pc.shape[0] != 3:
        pc = np.transpose(pc)

    assert pc.shape[0] == 3

    return np.concatenate((pc, np.ones((1, pc.shape[1]))), axis=0)


def get_average_metrics_over_vertebrae(result_list):
    """
    :param result_list: list of dict containing the losses for each vertebrae
    """

    average_dict = dict()
    for key in result_list[0].keys():
        average_dict[key] = np.nanmean([item[key] for item in result_list])

    print("avg_tre: ", average_dict["TRE"])

    return average_dict


def append_avg_metrics(result_list):
    average_dict = dict()
    for key in result_list[0].keys():
        if key == "id":
            average_dict[key] = "Average"
            continue
        average_dict[key] = np.mean([item[key] for item in result_list])

    result_list.append(average_dict)
    return result_list


def save_training_data(save_folder, data_id, source, deformed_source, original_flow, target, constraint, tre=None):
    gt_deformed_source = source + original_flow
    source = np.copy(deformed_source)
    new_flow = gt_deformed_source - deformed_source

    filename = os.path.join(save_folder, data_id + ".npz")

    np.savez_compressed(file=filename,
                        flow=new_flow,
                        pc1=source,
                        pc2=target,
                        ctsPts=constraint)

    if tre is not None:
        np.savetxt(os.path.join(save_folder, filename.split("_")[0] + "facet_targets.txt"), tre)


def run_cpd(data_batch, save_path, cpd_iterations=100, plot_iterations=False):
    # ##############################################################################################################
    # ############################################## Getting the data ##############################################
    # ##############################################################################################################
    source_pc, target_pc, color1, color2, gt_flow, mask1, constraint, position1, position2, file_name, tre_points \
        = data_batch
    constrain_pairs = get_connected_idxes(constraint)
    for i, item in enumerate(constrain_pairs):
        save_data(data_dict={'constraint_' + str(i): source_pc[item, ...]},
                  save_path=os.path.join(save_path, file_name))

    # Preprocessing and saving unprocessed data
    vertebra_dict = preprocess_input(source_pc, gt_flow, position1, constrain_pairs, tre_points)

    # ##############################################################################################################
    # ################################ 1.  1st CPD iteration on the full spine #####################################
    # ##############################################################################################################

    # 1.a First iteration to alight the spines
    cpd_method = BiomechanicalCpd(target_pc=target_pc, source_pc=source_pc, max_iterations=cpd_iterations)

    try:
        source_pc_it1, predicted_T_it1 = run_registration(cpd_method, with_callback=plot_iterations)
    except:
        return [{'mse loss': np.nan,
                 'Chamfer Distance': np.nan,
                 'translation distance': np.nan,
                 'quaternion distance': np.nan,
                 'TRE': np.nan}]
    # save_training_data(save_folder="E:/NAS/jane_project/pre_initialized_rigid_cpd",
    #                    data_id = file_name,
    #                    source = source_pc,
    #                    deformed_source = source_pc_it1,
    #                    original_flow= gt_flow,
    #                    target = target_pc,
    #                    constraint=constraint)

    # ##############################################################################################################
    # ################################ 2.  2nd CPD iteration on each vertebra ######################################
    # ##############################################################################################################

    # 2.a Getting the updated data to run the constrained CPD
    updated_source = source_pc_it1
    updated_gt_flow = source_pc + gt_flow - source_pc_it1  # the flow to move the source to target after iteration 1

    # 2.b Getting the updated pre-processed input data
    vertebra_dict_it1 = preprocess_input(updated_source, updated_gt_flow, position1, constrain_pairs, tre_points)

    # 2.c Iterate over all vertebrae and apply the constrained CPD
    result_iter2 = []

    full_source = []
    original_flow = []
    deformed_source = []
    tre_list = []
    for i, vertebra in enumerate(vertebra_dict_it1):

        # 2.d Selecting the target vertebra by proximity
        target_vertebra = get_closest_points(target_pc, vertebra['source'])

        # 2.e Running the constrained registration for the given vertebra
        reg = BiomechanicalCpd(target_pc=target_vertebra, source_pc=vertebra['source'], springs=vertebra['springs'],
                               max_iterations=cpd_iterations)

        try:
            source_pc_it2, predicted_T_it2 = run_registration(reg, with_callback=plot_iterations)
        except:
            return [{'mse loss': np.nan,
                     'Chamfer Distance': np.nan,
                     'translation distance': np.nan,
                     'quaternion distance': np.nan,
                     'TRE': np.nan}]
        deformed_source.append(source_pc_it2)
        full_source.append(vertebra_dict[i]['source'])
        original_flow.append(vertebra_dict[i]['gt_flow'])

        # 2.f Computing the overall transformation for the given vertebra
        original_source_vertebra = vertebra_dict[i]['source']
        homogenous_source = make_homogeneous(original_source_vertebra)
        overall_T = np.matmul(predicted_T_it2, predicted_T_it1)
        predicted_pc = np.matmul(overall_T, homogenous_source)
        predicted_pc = np.transpose(predicted_pc)[..., 0:3]

        if vertebra_dict[i]['tre_points'].size > 0:
            transformed_tre = np.matmul(overall_T, np.transpose(vertebra_dict[i]['tre_points']))
            tre_list.append(np.transpose(transformed_tre))

        # 2.g Sanity check using ground truth transform
        predicted_gt = np.matmul(vertebra_dict[i]['gt_transform'], homogenous_source)  # sanity check
        predicted_gt = np.transpose(predicted_gt)[..., 0:3]

        result_iter2.extend(get_result_dict(original_source_vertebra,
                                            vertebra_dict[i]['gt_flow'],
                                            predicted_pc,
                                            overall_T,
                                            tre_points=vertebra_dict[i]['tre_points'],
                                            position=None,
                                            ))

        # 2.h Saving data after second iteration
        print(f"{os.path.join(save_path, file_name)}, vertebra: L{i+1}")
        save_data(data_dict={'source_v' + str(i): original_source_vertebra,
                             'tre_points_v' + str(i): vertebra_dict[i]['tre_points'],
                             'target': target_pc,
                             'gt_flow_v' + str(i): vertebra_dict[i]['gt_flow'],
                             'predicted_pc_v' + str(i): predicted_pc,
                             'predicted_gt_v' + str(i): predicted_gt,
                             'moved_source_v' + str(i): source_pc_it2,  # sanity check
                             'predicted_T_v' + str(i): overall_T,
                             },
                  save_path=os.path.join(save_path, file_name))

    # save_training_data(save_folder="E:/NAS/jane_project/pre_initialized_cpd",
    #                    data_id = file_name,
    #                    source = np.concatenate(full_source, axis=0),
    #                    deformed_source = np.concatenate(deformed_source, axis=0),
    #                    original_flow= np.concatenate(original_flow, axis=0),
    #                    target = target_pc,
    #                    constraint=constraint,
    #                    tre = np.concatenate(tre_list, axis=0 if "spine22" in file_name else None)
    #                    )

    average_result = get_average_metrics_over_vertebrae(result_iter2)
    average_result['id'] = file_name

    return average_result


def main(dataset_path, save_path, cpd_iterations, rot_degree, rot_axis, wandb_key=None, wandb_name=""):
    wandb.login(key=wandb_key)
    wandb.init(project='spine_flownet_baseline')  # , mode = "disabled"
    wandb.run.name = wandb_name

    test_data_at = wandb.Artifact("test_samples_" + str(wandb.run.id), type="predictions")
    columns = ['id', 'mse loss', 'Chamfer Distance', 'quaternion distance', 'translation distance', 'TRE']
    test_table = wandb.Table(columns=columns)

    test_set = VerseFlowDataset(root=dataset_path, mode="test", raycasted=True, augment_test=rot_degree != 0,
                                test_rotation_degree=rot_degree, test_rotation_axis=rot_axis)

    results = []
    for i, data in enumerate(test_set):
        results.append(run_cpd(data_batch=data,
                               save_path=save_path,
                               cpd_iterations=cpd_iterations,
                               plot_iterations=False))

    results = append_avg_metrics(results)

    for data in results:
        table_entry = [data[item] for item in columns]
        test_table.add_data(*table_entry)

    test_data_at.add(test_table, "test prediction")
    wandb.run.log_artifact(test_data_at)
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data generation testing')
    # parser.add_argument('--dataset_path', type=str, default="./raycastedSpineClouds")
    parser.add_argument('--dataset_path', type=str, default="E:/NAS/jane_project/npz_data_raycasted")
    parser.add_argument('--wandb_key', type=str, required=True)
    parser.add_argument('--cpd-iterations', type=int, default=100)
    parser.add_argument('--save_path', type=str, default="./raycastedCPDRes")

    args = parser.parse_args()
    # for cpd_iterations in range(10, 100, 10):

    # main(dataset_path=args.dataset_path,
    #      save_path=args.save_path,
    #      cpd_iterations=20,
    #      rot_degree=0,
    #      rot_axis=None,
    #      wandb_key=args.wandb_key,
    #      wandb_name="cpd-no-spring-rot" + str(0))

    for axis in ["z"]:  #, "y", "x"
        for rotation in [-20]:
            main(dataset_path=args.dataset_path,
                 save_path=args.save_path,
                 cpd_iterations=20,
                 rot_degree=rotation,
                 rot_axis=axis,
                 wandb_key=args.wandb_key,
                 wandb_name="cpd-rot" + str(rotation) + "-axis-" + axis)
