import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class HLFeatExtractor(nn.Module):
    def __init__(self, input_dim, unit_dim=8):
        super(HLFeatExtractor, self).__init__()

        self.feat_extract = nn.Sequential(
            nn.Conv2d(input_dim, unit_dim * 12, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(unit_dim * 12),
            nn.ReLU(inplace=True),
            nn.Conv2d(unit_dim * 12, unit_dim * 12, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(unit_dim * 12),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(unit_dim * 12, unit_dim * 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(unit_dim * 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(unit_dim * 16, unit_dim * 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(unit_dim * 16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(unit_dim * 16, unit_dim * 20, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(unit_dim * 20),
            nn.ReLU(inplace=True),
            nn.Conv2d(unit_dim * 20, unit_dim * 20, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(unit_dim * 20),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        out = self.feat_extract(x)
        return out


class HierarchicalMultiScaleRegionLayer(nn.Module):
    def __init__(self, local_group, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, activation_type='ReLU'):
        super(HierarchicalMultiScaleRegionLayer, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, groups, bias)

        self.local_conv_branch1 = LocalConv2dReLU(local_group[0][0], local_group[0][1], out_channels, int(out_channels / 2),
                                                  kernel_size, stride,
                                                  padding, dilation, groups, bias, activation_type)
        self.local_conv_branch2 = LocalConv2dReLU(local_group[1][0], local_group[1][1], int(out_channels / 2),
                                                  int(out_channels / 4), kernel_size, stride,
                                                  padding, dilation, groups, bias, activation_type)
        self.local_conv_branch3 = LocalConv2dReLU(local_group[2][0], local_group[2][1], int(out_channels / 4),
                                                  int(out_channels / 4), kernel_size, stride,
                                                  padding, dilation, groups, bias, activation_type)

        self.bn = nn.BatchNorm2d(out_channels)

        if activation_type == 'ReLU':
            self.relu = nn.ReLU(inplace=True)
        elif activation_type == 'PReLU':
            self.relu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        local_branch1 = self.local_conv_branch1(x)
        local_branch2 = self.local_conv_branch2(local_branch1)
        local_branch3 = self.local_conv_branch3(local_branch2)
        local_out = torch.cat((local_branch1, local_branch2, local_branch3), 1)

        out = x + local_out
        out = self.bn(out)
        out = self.relu(out)

        return out


class HMRegionLearning(nn.Module):
    def __init__(self, input_dim=3, unit_dim=8):
        super(HMRegionLearning, self).__init__()

        self.multiscale_feat = nn.Sequential(
            HierarchicalMultiScaleRegionLayer([[8, 8], [4, 4], [2, 2]], input_dim, unit_dim * 4, kernel_size=3,
                                              stride=1, padding=1,
                                              activation_type='ReLU'),
            nn.MaxPool2d(kernel_size=2, stride=2),

            HierarchicalMultiScaleRegionLayer([[8, 8], [4, 4], [2, 2]], unit_dim * 4, unit_dim * 8, kernel_size=3,
                                              stride=1, padding=1,
                                              activation_type='ReLU'),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        multiscale_feat = self.multiscale_feat(x)
        return multiscale_feat