from monai.networks.nets import UNet, BasicUNet
from net_utils.utils import get_argparser_group
from typing import Sequence, Union
from torchvision.models.segmentation import fcn_resnet50
import pytorch_lightning as pl
import torch
from torch import nn
from collections import namedtuple


class FCNResNet50(pl.LightningModule):
    def __init__(self, hparams):
        super(FCNResNet50, self).__init__()
        self.hparams = hparams
        self.model = fcn_resnet50(pretrained=False, num_classes=1)

        self.model.backbone.conv1 = nn.Conv2d(in_channels=self.hparams.in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)

        print()

    def forward(self, x):
        out = self.model(x)['out']
        # if isinstance(out, dict):
        #     data_named_tuple = namedtuple("ModelEndpoints", sorted(out.keys()))  # type: ignore
        #     data = data_named_tuple(**out)  # type: ignore
        #
        # elif isinstance(out, list):
        #     data = tuple(out)

        return out

    @staticmethod
    def add_model_specific_args(parser):
        specific_args = get_argparser_group(title="Model options", parser=parser)
        specific_args.add_argument('--num_target_classes', type=int, default=2, help='num_target_classes')
        specific_args.add_argument('--image_width', type=int)
        specific_args.add_argument('--image_height', type=int)
        return parser












