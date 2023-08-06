import os
from datetime import datetime

import numpy as np
from torch import nn


class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def read_batch_data(data):

    if len(data) == 10:
        pc1, pc2, color1, color2, flow, mask1, constraint, position1, position2, file_name = data
    else:
        pc1, pc2, color1, color2, flow, mask1, constraint, position1, position2, file_name, tre_points = data

    pc1 = pc1.cuda().transpose(2, 1).contiguous().float()
    pc2 = pc2.cuda().transpose(2, 1).contiguous().float()
    color1 = color1.cuda().transpose(2, 1).contiguous().float()
    color2 = color2.cuda().transpose(2, 1).contiguous().float()
    flow = flow.cuda().transpose(2, 1).contiguous()
    mask1 = mask1.cuda().float()
    constraint = constraint.cuda()

    if len(data) == 10:
        return color1, color2, constraint, flow, pc1, pc2, position1, file_name

    return color1, color2, constraint, flow, pc1, pc2, position1, file_name, tre_points


def weights_init(m):
    classname=m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight.data)
        # torch.nn.init.constant(m.weight.data, 1/1000)
        # nn.init.xavier_normal(m.weight.data)
    if classname.find('Conv1d') != -1:
        nn.init.kaiming_normal_(m.weight.data)
        # torch.nn.init.constant(m.weight.data, 1/1000)
        # nn.init.xavier_normal(m.weight.data)


def create_paths(args):
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    os.makedirs(os.path.join(args.checkpoints_dir, 'models'), exist_ok=True)


def update_args(args):
    coeffs = []
    if isinstance(args.loss_coeff, float):
        args.loss_coeff = [args.loss_coeff]
    if isinstance(args.loss, str):
        args.loss = [args.loss]
    if len(args.loss) == 1:  # break using commas
        args.loss = args.loss[0].split(',')
    if len(args.loss_coeff) != 0:
        for coeff in args.loss_coeff:
            try:
                coeffs.append(float(coeff))
            except:
                raise Exception('loss coefficient should be a float number')
        assert len(coeffs) == len(args.loss), 'number of coefficients should be the same as the losses'
    else:
        coeffs = np.ones(len(args.loss))
    args.loss_coeff = {}
    for loss, coeff in zip(args.loss, coeffs):
        args.loss_coeff[loss] = coeff
    if args.coeff_rigidity is not None:
        args.loss_coeff['rigidity'] = args.coeff_rigidity
    if args.coeff_bio is not None:
        args.loss_coeff['biomechanical'] = args.coeff_bio
    if args.coeff_chamfer is not None:
        args.loss_coeff['chamfer'] = args.coeff_chamfer

    try:
        from polyaxon_client.tracking import Experiment
        args.checkpoints_dir = Experiment().get_outputs_path()
        if '/mnt' in args.dataset_path:
            args.dataset_path = args.dataset_path.replace('/mnt/polyaxon/data1/', '/data/')
        print("You are running on the cluster :)")
        print(args)
    except Exception as e:
        print(e)
        now = datetime.now()
        args.checkpoints_dir = os.path.join('checkpoints/', 'flownet3d/', f'{now.strftime("%m.%d.%Y_%H.%M.%S")}/')
        print("You are Running on the local Machine")
        print(args)

    if args.test_output_path is None:
        args.test_output_path = args.checkpoints_dir

    if args.test_dataset_path is None:
        args.test_dataset_path = args.dataset_path

    return args


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)