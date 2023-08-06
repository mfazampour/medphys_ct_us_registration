from monai.networks.nets import UNet, BasicUNet
from net_utils.utils import get_argparser_group
from typing import Sequence, Union
from torchvision.models import resnet18
import pytorch_lightning as pl
import torch
from torch import nn


class ResNet(pl.LightningModule):
    def __init__(self, hparams):
        super(ResNet, self).__init__()
        self.model = resnet18()

        self.model.conv1 = nn.Conv2d(in_channels=hparams.input_channels, out_channels=64, kernel_size=7,
                              stride=2, padding=3, bias=False)
        num_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_features, hparams.num_target_classes)

        # parameters
        self.hparams = hparams

        # # init a pretrained resnet
        # layers = list(self.model.children())[:-1]
        # layers[0] = nn.Conv2d(in_channels=hparams.input_channels, out_channels=layers[0].out_channels, kernel_size=7,
        #                       stride=2, padding=3, bias=False)
        # self.feature_extractor = torch.nn.Sequential(*layers)
        #
        # dummy_tensor = torch.zeros((1, 1, hparams.image_width, hparams.image_height))
        # out = self.feature_extractor(dummy_tensor)
        # out = out.view(out.shape[0], -1)
        # # use the pretrained model to classify cifar-10 (10 image classes)
        # self.classifier = [nn.Linear(out.shape[1], hparams.num_target_classes)]
        # self.classifier = nn.Sequential(*self.classifier)

    def forward(self, x):
        out = self.model(x)
        return out

    @staticmethod
    def add_model_specific_args(parser):
        specific_args = get_argparser_group(title="Model options", parser=parser)
        specific_args.add_argument('--num_target_classes', type=int, default=2, help='num_target_classes')
        specific_args.add_argument('--image_width', type=int)
        specific_args.add_argument('--image_height', type=int)
        return parser
