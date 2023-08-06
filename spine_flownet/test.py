from __future__ import print_function

import datetime
import os

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

import utils
from test_utils import *
from data import SceneflowDataset, VerseFlowDataset
from model import FlowNet3D, FlowNet3DLegacy


def tensor2numpy(*args):
    for item in args:
        if isinstance(item, torch.Tensor):
            item = item.to("cpu").numpy()
        yield item


def compute_test_metrics(file_id, source_pc, source_color, gt_flow, estimated_flow, tre_points):
    """
    :param: file_id: str: The id of the file
    :param: source_pc: np.ndarray() with size [n_batches, 3, n_points] containing the source points
    :param: source_color: np.ndarray() with size [n_batches, 1, n_points] containing the source points color
    :param: gt_flow: np.ndarray() with size [n_batches, 3, n_points] containing the ground truth flow
    :param: estimated_flow: np.ndarray() with size [n_batches, 3, n_points] containing the estimated flow
    """

    # todo this is a fix not to make code crash - todo is adding tre points also for validation data
    if tre_points is None:
        tre_points = np.ones((source_pc.shape[0], 2, 4)) * -1

    source_pc, source_color, gt_flow, estimated_flow, tre_points = \
        tensor2numpy(source_pc, source_color, gt_flow, estimated_flow, tre_points)

    # concatenating source_pc and color such that points color is stored in the fourth column of the source_pc
    source_pc = np.concatenate((source_pc, np.expand_dims(source_color, 1)), axis=1)

    batch_size = source_pc.shape[0]

    q_dist_batch = 0
    tr_dist_batch = 0
    tre_batch = 0
    diff_tre_batch = 0
    metric_dict_list = []
    for i in range(batch_size):
        # error in terms of quaternion distance and translation distance. Point clouds are transposed as the function
        # expects [n_points x 3] arrays and not [3 x n_points] arrays.
        quaternion_distance_list, translation_distance_list, tre, diff_tre = vertebrae_pose_error(
            source=np.transpose(source_pc[i, ...]),
            gt_flow=np.transpose(gt_flow[i, ...]),
            predicted_flow=np.transpose(estimated_flow[i, ...]),
            tre_points=tre_points[i, ...])

        metric_dict_list.append({
            "id": file_id,
            "quaternion distance": np.mean(quaternion_distance_list),
            "translation distance": np.mean(translation_distance_list),
            "TRE": np.mean(tre),
            "init_TRE": np.mean(diff_tre)
        })
        q_dist_batch += np.mean(quaternion_distance_list)
        tr_dist_batch += np.mean(translation_distance_list)
        tre_batch += np.mean(tre)
        diff_tre_batch += np.mean(diff_tre)

    return metric_dict_list, q_dist_batch / batch_size, tr_dist_batch / batch_size, tre_batch / batch_size, diff_tre_batch / batch_size


def save_metrics2csv(metric_dict_list):
    pass


def log_metrics_dict2wandb(metric_dict_list, wandb_table):
    table_columns = wandb_table.columns

    # Adding the average value over the whole test data set
    metric_dict_list.append({
        "id": "Average",
        "quaternion distance": np.mean([item["quaternion distance"] for item in metric_dict_list]),
        "translation distance": np.mean([item["translation distance"] for item in metric_dict_list]),
        "mse loss": np.mean([item["mse loss"] for item in metric_dict_list]),
        "biomechanical loss": np.mean([item["biomechanical loss"] for item in metric_dict_list]),
        "rigidity loss": np.mean([item["rigidity loss"] for item in metric_dict_list]),
        "Chamfer loss": np.mean([item["Chamfer loss"] for item in metric_dict_list]),
        "TRE": np.mean([item["TRE"] for item in metric_dict_list])
    })

    # Logging to wandb
    for data in metric_dict_list:
        table_entry = [data[item] for item in table_columns]
        wandb_table.add_data(*table_entry)


def save_data(save_path, file_id, source_pc, source_color, target_pc, predicted_pc, gt_pc):
    os.makedirs(save_path, exist_ok=True)
    source_pc, target_pc, predicted_pc, gt_pc = tensor2numpy(source_pc, target_pc, predicted_pc, gt_pc)
    batch_size = source_pc.shape[0]

    # concatenating the source color to the point clouds. As the target_pc and the predicted_pc are computed as
    # source + flow, the point clouds are assumed to be corresponding
    source_pc = np.concatenate((source_pc, np.expand_dims(source_color, 1)), axis=1)
    target_pc = np.concatenate((target_pc, np.zeros((batch_size, 1, source_color.size))), axis=1)
    predicted_pc = np.concatenate((predicted_pc, np.expand_dims(source_color, 1)), axis=1)
    gt_pc = np.concatenate((gt_pc, np.expand_dims(source_color, 1)), axis=1)

    for i in range(batch_size):
        np.savetxt(os.path.join(save_path, f"source_{file_id[i]}.txt"), np.transpose(source_pc[i, ...]))
        np.savetxt(os.path.join(save_path, f"gt_{file_id[i]}.txt"), np.transpose(gt_pc[i, ...]))
        np.savetxt(os.path.join(save_path, f"target_{file_id[i]}.txt"), np.transpose(target_pc[i, ...]))
        np.savetxt(os.path.join(save_path, f"predicted_{file_id[i]}.txt"), np.transpose(predicted_pc[i, ...]))


def get_color_array(vertebrae_idxs):
    """
    According to the data loader, vertebrae_idxs is a list of 5 tensors with size [n_batches, npoints]
    """

    batch_size = vertebrae_idxs[0].size(0)
    n_points = np.sum([item.size(1) for item in vertebrae_idxs])

    color_array = np.zeros((batch_size, n_points))
    for i, vertebra_idxes in enumerate(vertebrae_idxs):
        vertebra_idxes, = tensor2numpy(vertebra_idxes)

        for batch_n in range(vertebra_idxes.shape[0]):
            color_array[batch_n, vertebra_idxes[batch_n]] = i + 1

    return color_array


def test_one_epoch(net, test_loader, args, save_results=False, wandb_table: wandb.Table = None, max_num_batch=-1):
    net.eval()
    total_loss = 0
    mse_loss_total, bio_loss_total, rig_loss_total, chamfer_loss_total, tre_total, diff_tre_total = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    quaternion_distance_total, translation_distance_total = 0.0, 0.0

    # Initializing the save folder if save_results is set to True
    result_path = os.path.join(args.test_output_path, args.exp_name, 'test_result/')
    if save_results:
        os.makedirs(result_path, exist_ok=True)

    test_metrics = []
    timespans = []
    for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
        if max_num_batch != -1 and i >= max_num_batch:
            break
        batch_data = utils.read_batch_data(data)
        start_time = datetime.datetime.now()
        if len(batch_data) == 9:
            color1, color2, constraint, flow, pc1, pc2, position1, fn, tre_points = batch_data
        else:
            color1, color2, constraint, flow, pc1, pc2, position1, fn = batch_data
            tre_points = None
        source_color = get_color_array(position1)

        batch_size = pc1.size(0)
        flow_pred = net(pc1, pc2, color1, color2)
        bio_loss, chamfer_loss, loss, mse_loss, rig_loss = utils.calculate_loss(batch_size, constraint, flow, flow_pred,
                                                                                ['all'], pc1, pc2, position1,
                                                                                args.loss_coeff.copy())

        timespans.append((datetime.datetime.now() - start_time).total_seconds())
        metrics, quaternion_distance, translation_distance, tre, diff_tre = compute_test_metrics(file_id=fn,
                                                                                                 source_pc=pc1,
                                                                                                 source_color=source_color,
                                                                                                 gt_flow=flow,
                                                                                                 estimated_flow=flow_pred,
                                                                                                 tre_points=tre_points)
        for item in metrics:
            item["mse loss"] = mse_loss.item()
            item["biomechanical loss"] = bio_loss.item()
            item["rigidity loss"] = rig_loss.item()
            item["Chamfer loss"] = chamfer_loss.item()

        test_metrics.extend(metrics)
        # Adding the network losses to the losses list of dicts

        mse_loss_total += mse_loss.item() / len(test_loader)
        bio_loss_total += bio_loss.item() / len(test_loader)
        rig_loss_total += rig_loss.item() / len(test_loader)
        chamfer_loss_total += chamfer_loss.item() / len(test_loader)
        quaternion_distance_total += quaternion_distance / len(test_loader)
        translation_distance_total += translation_distance / len(test_loader)
        tre_total += tre / len(test_loader)
        diff_tre_total += diff_tre / len(test_loader)

        total_loss += loss.item() / len(test_loader)

        if save_results and args.wandb_sweep_id is None:
            # Saving the point clouds
            save_data(save_path=result_path,
                      file_id=fn,
                      source_pc=pc1,
                      source_color=source_color,
                      target_pc=pc2,
                      predicted_pc=pc1 + flow_pred,
                      gt_pc=pc1 + flow)

            # plotting on wandb
            for j in range(test_loader.batch_size):
                utils.plot_pointcloud(flow_pred[j:j + 1, ...], pc1[j:j + 1, ...], pc2[j:j + 1, ...], tag=fn[j], mode='test')

    if wandb_table is not None:
        log_metrics_dict2wandb(test_metrics, wandb_table)

    losses = {'total_loss': total_loss, 'mse_loss': mse_loss_total, 'biomechanical_loss': bio_loss_total,
              'rigid_loss': rig_loss_total, 'chamfer_loss': chamfer_loss_total,
              'quaternion_distance': quaternion_distance_total, 'translation_distance': translation_distance_total,
              'TRE': tre_total, 'TRE_diff': diff_tre_total}

    print(f'test duration is {np.mean(timespans)}')

    return losses


def create_output_paths(output_path, exp_name):
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, exp_name), exist_ok=True)

    return output_path


def main():
    parser = utils.create_parser()
    args = parser.parse_args()

    args = utils.update_args(args)

    assert os.path.exists(args.model_path), f'model path {args.model_path} does not exist'

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # boardio = SummaryWriter(log_dir='checkpoints/' + args.exp_name)
    boardio = []

    args.test_output_path = create_output_paths(args.test_output_path, args.exp_name)

    wandb.login(key=args.wandb_key)
    wandb.init(project='spine_flownet', config=args)  # , mode = "disabled"

    textio = utils.IOStream(args.test_output_path + '/run.log')
    textio.cprint(str(args))

    if args.no_legacy_model:
        net = FlowNet3D(args).cuda()
    else:
        net = FlowNet3DLegacy(args).cuda()
    net.load_state_dict(torch.load(args.model_path))
    net.eval()

    test(args, net, textio)


def test(args, net, textio, spine_splits=None):
    test_set = SceneflowDataset(npoints=4096, mode="test", root=args.test_dataset_path,
                                raycasted=args.use_raycasted_data, data_seed=args.data_seed,
                                test_id=args.test_id, splits=spine_splits,
                                max_rotation=args.max_rotation, augment_test=args.augment_test,
                                test_rotation_axis=args.test_rotation_axis, test_rotation_degree=args.max_rotation,
                                occlude_data=args.occlude_data, occlude_ratio=args.occlude_ratio)
    test_loader = DataLoader(test_set, batch_size=1, drop_last=False, num_workers=args.num_workers)

    test_data_at = wandb.Artifact("test_samples_" + str(wandb.run.id), type="predictions")

    columns = ['id', "mse loss", "biomechanical loss", "Chamfer loss", 'rigidity loss',
               'quaternion distance', 'translation distance', 'TRE']
    test_table = wandb.Table(columns=columns)

    with torch.no_grad():
        test_loss = test_one_epoch(net, test_loader, args=args, save_results=True, wandb_table=test_table)

    wandb.log({'Test': test_loss})

    textio.cprint('==FINAL TEST==')
    textio.cprint(f'mean test loss: {test_loss}')

    test_data_at.add(test_table, "test prediction")
    wandb.run.log_artifact(test_data_at)


def test_verse20(args, net, textio, spine_splits=None):
    test_set = VerseFlowDataset(npoints=4096, root=args.test_dataset_path, mode="test",
                                raycasted=args.use_raycasted_data, data_seed=args.data_seed, test_id=args.test_id,
                                splits=spine_splits, occlude_data=args.occlude_data, occlude_ratio=args.occlude_ratio,
                                max_rotation=args.max_rotation, augment_test=args.augment_test,
                                test_rotation_axis=args.test_rotation_axis, test_rotation_degree=args.max_rotation)
    test_loader = DataLoader(test_set, batch_size=1, drop_last=False, num_workers=args.num_workers)

    test_data_at = wandb.Artifact("test_samples_" + str(wandb.run.id), type="predictions")

    columns = ['id', "mse loss", "biomechanical loss", "Chamfer loss", 'rigidity loss',
               'quaternion distance', 'translation distance', 'TRE']
    test_table = wandb.Table(columns=columns)

    with torch.no_grad():
        test_loss = test_one_epoch(net, test_loader, args=args, save_results=True, wandb_table=test_table)

    wandb.log({'Test': test_loss})

    textio.cprint('==FINAL TEST==')
    textio.cprint(f'mean test loss: {test_loss}')

    test_data_at.add(test_table, "test prediction")
    wandb.run.log_artifact(test_data_at)


if __name__ == "__main__":
    main()
