import torch.nn as nn
import torch.nn.functional as F
import torch

from utils import PointNetSetAbstraction, PointNetFeaturePropogation, FlowEmbedding, PointNetSetUpConv
from utils import create_parser

import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


class FlowNet3D(nn.Module):
    def __init__(self,args):
        super(FlowNet3D, self).__init__()

        RADIUS1 = 5.0
        RADIUS2 = 10.0
        RADIUS3 = 17.5
        RADIUS4 = 25.0

        num_filt = args.num_filt

        n_points = args.num_points
        n_points_denominator = args.n_points_denominator

        self.sa1 = PointNetSetAbstraction(npoint=n_points, radius=RADIUS1, nsample=16, in_channel=3, mlp=[num_filt//2, num_filt//2, num_filt], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=n_points//n_points_denominator[0], radius=RADIUS2, nsample=16, in_channel=num_filt, mlp=[num_filt, num_filt, num_filt * 2], group_all=False)

        self.sa3 = PointNetSetAbstraction(npoint=n_points//n_points_denominator[1], radius=RADIUS3, nsample=8, in_channel=num_filt * 2, mlp=[num_filt * 2, num_filt * 2, num_filt * 4], group_all=False)
        self.sa4 = PointNetSetAbstraction(npoint=n_points//n_points_denominator[2], radius=RADIUS4, nsample=8, in_channel=num_filt * 4, mlp=[num_filt * 4, num_filt * 4, num_filt * 8], group_all=False)

        self.fe_layer = FlowEmbedding(radius=10.0, nsample=64, in_channel=num_filt * 2, mlp=[num_filt * 2, num_filt * 2, num_filt * 2], pooling='max', corr_func='concat', knn=True)

        self.su1 = PointNetSetUpConv(nsample=8, radius=24, f1_channel=num_filt * 4, f2_channel=num_filt * 8, mlp=[], mlp2=[num_filt * 4, num_filt * 4], knn=True)
        self.su2 = PointNetSetUpConv(nsample=8, radius=12, f1_channel=num_filt * 4, f2_channel=num_filt * 4, mlp=[num_filt * 2, num_filt * 2, num_filt * 4], mlp2=[num_filt * 4], knn=True)
        self.su3 = PointNetSetUpConv(nsample=8, radius=6, f1_channel=num_filt, f2_channel=num_filt * 4, mlp=[num_filt * 2, num_filt * 2, num_filt * 4], mlp2=[num_filt * 4], knn=True)
        self.fp = PointNetFeaturePropogation(in_channel=num_filt * 4 + 3, mlp=[num_filt * 4, num_filt * 4])

        self.conv1 = nn.Conv1d(num_filt * 4, num_filt * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(num_filt * 2)
        self.conv2 = nn.Conv1d(num_filt * 2, 3, kernel_size=1, bias=True)
        
    def forward(self, pc1, pc2, feature1, feature2):
        l1_pc1, l1_feature1 = self.sa1(pc1, feature1)
        l2_pc1, l2_feature1 = self.sa2(l1_pc1, l1_feature1)
        
        l1_pc2, l1_feature2 = self.sa1(pc2, feature2)
        l2_pc2, l2_feature2 = self.sa2(l1_pc2, l1_feature2)
        
        _, l2_feature1_new = self.fe_layer(l2_pc1, l2_pc2, l2_feature1, l2_feature2)

        l3_pc1, l3_feature1 = self.sa3(l2_pc1, l2_feature1_new)
        l4_pc1, l4_feature1 = self.sa4(l3_pc1, l3_feature1)
        
        l3_fnew1 = self.su1(l3_pc1, l4_pc1, l3_feature1, l4_feature1)
        l2_fnew1 = self.su2(l2_pc1, l3_pc1, torch.cat([l2_feature1, l2_feature1_new], dim=1), l3_fnew1)
        l1_fnew1 = self.su3(l1_pc1, l2_pc1, l1_feature1, l2_fnew1)
        l0_fnew1 = self.fp(pc1, l1_pc1, feature1, l1_fnew1)
        
        x = F.relu(self.bn1(self.conv1(l0_fnew1)))
        sf = self.conv2(x)
        return sf


class FlowNet3DLegacy(nn.Module):
    def __init__(self, args):
        super(FlowNet3DLegacy, self).__init__()

        self.num_points = args.num_points

        self.sa1 = PointNetSetAbstraction(npoint=1024, radius=5, nsample=16, in_channel=3, mlp=[32, 32, 64],
                                          group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=256, radius=10, nsample=16, in_channel=64, mlp=[64, 64, 128],
                                          group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=64, radius=17.5, nsample=8, in_channel=256, mlp=[128, 128, 256],
                                          group_all=False)
        self.sa4 = PointNetSetAbstraction(npoint=16, radius=25, nsample=8, in_channel=256, mlp=[256, 256, 512],
                                          group_all=False)

        self.fe_layer = FlowEmbedding(radius=2.5, nsample=64, in_channel=128, mlp=[256], pooling='max',
                                      corr_func='concat')

        self.su1 = PointNetSetUpConv(nsample=8, radius=24, f1_channel=256, f2_channel=512, mlp=[], mlp2=[256, 256])
        self.su2 = PointNetSetUpConv(nsample=8, radius=12, f1_channel=3 * 128, f2_channel=256, mlp=[128, 128, 256],
                                     mlp2=[256])
        self.su3 = PointNetSetUpConv(nsample=8, radius=6, f1_channel=64, f2_channel=256, mlp=[128, 128, 256],
                                     mlp2=[256])
        self.fp = PointNetFeaturePropogation(in_channel=256 + 3, mlp=[256, 256])
        # self.num_points = args.num_points
        #
        # self.sa1 = PointNetSetAbstraction(npoint=1024, radius=0.5, nsample=16, in_channel=3, mlp=[ 64],
        #                                   group_all=False)
        # self.sa2 = PointNetSetAbstraction(npoint=256, radius=1.0, nsample=16, in_channel=64, mlp=[ 128],
        #                                   group_all=False)
        # self.sa3 = PointNetSetAbstraction(npoint=64, radius=1.75, nsample=8, in_channel=256, mlp=[256],
        #                                   group_all=False)
        # self.sa4 = PointNetSetAbstraction(npoint=16, radius=2.5, nsample=8, in_channel=256, mlp=[ 512],
        #                                   group_all=False)
        #
        # self.fe_layer = FlowEmbedding(radius=.25, nsample=64, in_channel=128, mlp=[256], pooling='max',
        #                               corr_func='concat')
        #
        # self.su1 = PointNetSetUpConv(nsample=8, radius=2.4, f1_channel=256, f2_channel=512, mlp=[], mlp2=[256])
        # self.su2 = PointNetSetUpConv(nsample=8, radius=1.2, f1_channel=3 * 128, f2_channel=256, mlp=[256],
        #                              mlp2=[256])
        # self.su3 = PointNetSetUpConv(nsample=8, radius=0.6, f1_channel=64, f2_channel=256, mlp=[ 256],
        #                              mlp2=[256])
        # self.fp = PointNetFeaturePropogation(in_channel=256 + 3, mlp=[256, 256])
        #
        self.conv1 = nn.Conv1d(256, 128, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 3, kernel_size=1, bias=True)

    def forward(self, pc1, pc2, feature1, feature2):
        l1_pc1, l1_feature1 = self.sa1(pc1, feature1)
        l2_pc1, l2_feature1 = self.sa2(l1_pc1, l1_feature1)

        l1_pc2, l1_feature2 = self.sa1(pc2, feature2)
        l2_pc2, l2_feature2 = self.sa2(l1_pc2, l1_feature2)

        _, l2_feature1_new = self.fe_layer(l2_pc1, l2_pc2, l2_feature1, l2_feature2)

        l3_pc1, l3_feature1 = self.sa3(l2_pc1, l2_feature1_new)
        l4_pc1, l4_feature1 = self.sa4(l3_pc1, l3_feature1)

        l3_fnew1 = self.su1(l3_pc1, l4_pc1, l3_feature1, l4_feature1)
        l2_fnew1 = self.su2(l2_pc1, l3_pc1, torch.cat([l2_feature1, l2_feature1_new], dim=1), l3_fnew1)
        l1_fnew1 = self.su3(l1_pc1, l2_pc1, l1_feature1, l2_fnew1)
        l0_fnew1 = self.fp(pc1, l1_pc1, feature1, l1_fnew1)

        x = F.relu(self.bn1(self.conv1(l0_fnew1)))
        sf = self.conv2(x)
        return sf


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

if __name__ == '__main__':
    import os
    import data
    parser = create_parser()
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    d = data.SceneflowDataset()
    pc1, pc2, color1, color2, flow, mask1 = d[0]
    # label = torch.randn(8,16)
    model = FlowNet3D(args)
    print(model)

