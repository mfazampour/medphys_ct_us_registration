"""
See this link for the idea behind the UNet:
https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
UNet example is adapted from:
https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
"""

import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
from net_utils.utils import get_argparser_group


class UNet2D(pl.LightningModule):
    def __init__(self, hparams):
        super(UNet2D, self).__init__()
        # parameters
        self.hparams = hparams
        self.n_channels = hparams.in_channels
        self.n_classes = hparams.out_channels
        self.bilinear = hparams.bilinear
        # define the actual network architecture
        self.inc = DoubleConv(self.n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, self.bilinear)
        self.up2 = Up(256, 128 // factor, self.bilinear)
        self.up3 = Up(128, 64 // factor, self.bilinear)
        self.up4 = Up(64, 32, self.bilinear)
        self.out = nn.Conv2d(32, self.n_classes, kernel_size=1)
        # final layer for activation i.e. converting the logits / predictions to a value between 0 and 1
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # batch, channels, width, height = x.size()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        y_pred = self.out(x)
        #y_pred = self.activation(x)

        #y_pred = x
        # y_pred will contain probabilities between 0 and 1
        # and is of shape batch_size, channels, x, y
        # for 2 classes and batch size 1: 1, 2, x, y
        # The probabilities along a pixel must sum up to 1
        # For example, if y_pred[0, 0, 0, 0] = .3, then y_pred[0, 1, 0, 0] must be .7 (if there are only two classes)
        return y_pred

    @staticmethod
    def add_model_specific_args(parser):
        specific_args = get_argparser_group(title="Model options", parser=parser)
        specific_args.add_argument('--bilinear', default=True, type=bool, help='')
        return parser


class DoubleConv(pl.LightningModule):
    """
    First step of the network: apply 3x3 conv + ReLU twice to the input image of size 572 * 572 * 1.
    The output size of this first step is 568 * 568 * 64.
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        firstconv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1)
        return self.double_conv(x.float())


class Down(pl.LightningModule):
    """
    Each down block starts off with a 2x2 max pooling operation with stride 2
    and then contains two 3x3 convolutions + ReLu.
    With each down block, the image size is first halved and then reduced by 4 while the number of channels is doubled
    (e.g. 568 * 568 * 64 will become 280 * 280 * 126 where 280 = 568 / 2 - 4
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(pl.LightningModule):
    """
    Each up block starts off with a 2x2 up convolution with stride 2
    and then applies a copy and crop operation using the skip connections i.e. the data from the down path
    before applying two 3x3 convolutions + ReLu.
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
