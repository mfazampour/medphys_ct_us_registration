#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import copy
import datetime
import os
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from data import SceneflowDataset
from model import FlowNet3D
from test import test_one_epoch, test, compute_test_metrics, get_color_array

args = None

def train(args, net, train_loader, val_loader, test_loader, textio):
    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(net.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-5)
    # scheduler = MultiStepLR(opt, milestones=[75, 150, 200], gamma=0.1)
    scheduler = StepLR(opt, 10, gamma=0.7)

    best_test_loss = np.inf
    best_net = None
    report_val_loss = False
    for epoch in range(args.epochs):
        textio.cprint('==epoch: %d, learning rate: %f==' % (epoch, opt.param_groups[0]['lr']))
        train_losses = train_one_epoch(net, train_loader, opt, args.loss, args)
        textio.cprint('mean train EPE loss: %f' % train_losses['total_loss'])

        with torch.no_grad():
            val_losses = test_one_epoch(net, val_loader, args=args, wandb_table=None, max_num_batch=-1)
            test_loss = test_one_epoch(net, test_loader, args=args, save_results=False, wandb_table=None)
        val_loss = val_losses['TRE']
        textio.cprint('mean test loss: %f' % val_loss)
        if best_test_loss >= val_loss:
            best_test_loss = val_loss
            report_val_loss = True
            best_net = copy.deepcopy(net)
            textio.cprint('best test loss till now: %f' % val_loss)
            if torch.cuda.device_count() > 1 and args.gpu_id == -1:
                torch.save(net.module.state_dict(), f'{args.checkpoints_dir}/models/model_spine_bio.best.t7')
            else:
                torch.save(net.state_dict(), f'{args.checkpoints_dir}/models/model_spine_bio.best.t7')

        scheduler.step()

        if report_val_loss:
            wandb.log({'Train': train_losses, 'Validation': val_losses,
                       'val_loss': val_losses[args.sweep_target_loss], 'Test during training': test_loss})
            report_val_loss = False
        else:
            wandb.log({'Train': train_losses, 'Validation': val_losses, 'Test during training': test_loss})

        args.lr = scheduler.get_last_lr()[0]
    return best_net


def train_one_epoch(net, train_loader, opt, loss_opt, args):
    net.train()
    total_loss = 0
    mse_loss_total, bio_loss_total, rig_loss_total, chamfer_loss_total, tre_total, diff_tre_total = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        batch_data = utils.read_batch_data(data)
        if len(batch_data) == 9:
            color1, color2, constraint, flow, pc1, pc2, position1, fn, tre_points = batch_data
        else:
            color1, color2, constraint, flow, pc1, pc2, position1, fn = batch_data
            tre_points = None
        source_color = get_color_array(position1)
        batch_size = pc1.size(0)
        opt.zero_grad()
        flow_pred = net(pc1, pc2, color1, color2)
        bio_loss, chamfer_loss, loss, mse_loss, rig_loss = utils.calculate_loss(batch_size, constraint, flow, flow_pred,
                                                                                loss_opt, pc1, pc2, position1,
                                                                                args.loss_coeff)
        loss.backward()
        opt.step()
        with torch.no_grad():
            metrics, quaternion_distance, translation_distance, tre, diff_tre = compute_test_metrics(file_id=fn,
                                                                                       source_pc=pc1,
                                                                                       source_color=source_color,
                                                                                       gt_flow=flow,
                                                                                       estimated_flow=flow_pred.detach(),
                                                                                       tre_points=tre_points)

            mse_loss_total += mse_loss.item() / len(train_loader)
            bio_loss_total += bio_loss.item() / len(train_loader)
            rig_loss_total += rig_loss.item() / len(train_loader)
            chamfer_loss_total += chamfer_loss.item() / len(train_loader)
            total_loss += loss.item() / len(train_loader)
            tre_total += tre / len(train_loader)
            diff_tre_total += diff_tre / len(train_loader)

        if i % 50 == 0 and args.wandb_sweep_id is None:  # plot only if not in sweep mode
            utils.plot_pointcloud(flow_pred, pc1, pc2)

    losses = {'total_loss': total_loss, 'mse_loss': mse_loss_total, 'biomechanical_loss': bio_loss_total,
              'rigid_loss': rig_loss_total, 'chamfer_loss': chamfer_loss_total, 'TRE': tre_total, 'TRE_diff': diff_tre_total}
    return losses


def run_experiment(args):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(100)
    torch.cuda.manual_seed_all(100)
    np.random.seed(100)
    # np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)
    utils.create_paths(args)
    textio = utils.IOStream(os.path.join(args.checkpoints_dir, 'run.log'))
    textio.cprint(str(args))
    if args.gpu_id != -1:
        torch.cuda.set_device(args.gpu_id)
    net = FlowNet3D(args).cuda()
    net.apply(utils.weights_init)

    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print(f'%%% number of parameters: {utils.count_parameters(net):.3E}')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

    train_set = SceneflowDataset(npoints=4096, mode="train", root=args.dataset_path,
                                 raycasted=args.use_raycasted_data, augment=not args.no_augmentation,
                                 data_seed=args.data_seed, test_id=args.test_id,
                                 max_rotation=args.max_rotation, train_set_size=args.train_set_size,
                                 occlude_data=args.occlude_data, occlude_ratio=args.occlude_ratio)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers)

    print(train_set.spine_splits)
    val_set = SceneflowDataset(npoints=4096, mode="val", root=args.dataset_path,
                               raycasted=args.use_raycasted_data, splits=train_set.spine_splits,
                               max_rotation=args.max_rotation, augment_test=args.augment_test,
                               occlude_data=args.occlude_data, occlude_ratio=args.occlude_ratio)
    val_loader = DataLoader(val_set, batch_size=1, drop_last=False, num_workers=args.num_workers)

    test_set = SceneflowDataset(npoints=4096, mode="test", root=args.test_dataset_path,
                                raycasted=args.use_raycasted_data, data_seed=args.data_seed,
                                test_id=args.test_id, splits=train_set.spine_splits,
                                max_rotation=args.max_rotation, augment_test=args.augment_test,
                                test_rotation_axis=args.test_rotation_axis, test_rotation_degree=args.max_rotation,
                                occlude_data=args.occlude_data, occlude_ratio=args.occlude_ratio)
    test_loader = DataLoader(test_set, batch_size=1, drop_last=False, num_workers=args.num_workers)

    if torch.cuda.device_count() > 1 and args.gpu_id == -1:
        net = nn.DataParallel(net)
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    best_net = train(args, net, train_loader, val_loader, test_loader, textio)
    # test after training
    test(args, best_net, textio, spine_splits=train_set.spine_splits)


def train_wandb():
    global args
    args_ = copy.deepcopy(args)
    with wandb.init(project='spine_flownet', config=args_):
        config = wandb.config
        args_ = SimpleNamespace(**config)
        args_ = utils.update_args(args_)
        print('-------------------config---------------------')
        print(args_)

        run_experiment(args_)



def main():
    global args
    parser = utils.create_parser()
    args = parser.parse_args()

    wandb.login(key=args.wandb_key)

    if args.wandb_sweep_id is not None:
        wandb.agent(args.wandb_sweep_id, train_wandb, count=args.wandb_sweep_count, project='spine_flownet')
    else:
        args = utils.update_args(args)

        wandb.init(project='spine_flownet', config=args)

        run_experiment(args)


if __name__ == '__main__':
    main()
