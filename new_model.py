import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# ldmark_49 = [(23,26),(5,6),(2,9),(3,8),(25,30),(22,27),(44,46),(32,38),(40,42),(35,41)]
# ldmark_68 = [(40,43),(22,23),(19,26),(20,25),(42,47),(39,44),(62,64),(49,55),(57,59),(52,58)]

class HLFeatExtractor(nn.Module):
    def __init__(self, input_dim, unit_dim=8):
        super(HLFeatExtractor, self).__init__()

        self.feat_extract = nn.Sequential(
            nn.Conv2d(input_dim, unit_dim * 12, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.BatchNorm2d(unit_dim * 12),
            nn.ReLU(inplace=True),
            nn.Conv2d(unit_dim * 12, unit_dim * 12, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.BatchNorm2d(unit_dim * 12),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(unit_dim * 12, unit_dim * 16, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.BatchNorm2d(unit_dim * 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(unit_dim * 16, unit_dim * 16, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.BatchNorm2d(unit_dim * 16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(unit_dim * 16, unit_dim * 20, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.BatchNorm2d(unit_dim * 20),
            nn.ReLU(inplace=True),
            nn.Conv2d(unit_dim * 20, unit_dim * 20, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.BatchNorm2d(unit_dim * 20),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        out = self.feat_extract(x)
        return out


class LocalConv2dReLU(nn.Module):
    """
    The output feature is divided into 8x8 patches (or 4x4 or 2x2).
    For each of the patch we designed a set of convolutional layer, with batch normalization and RELU, 
    Thus we will have 64 sets for 8x8 patch layer, and 16 sets for 4x4 patch layer.
    """

    def __init__(self, local_h_num, local_w_num, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, activation_type='ReLu'):
        super(LocalConv2dReLU, self).__init__()
        self.local_h_num = local_h_num  # number of patches for height
        self.local_w_num = local_w_num  # number of patches for width

        self.bns = nn.ModuleList([nn.BatchNorm2d(in_channels)
                                  for i in range(local_h_num*local_w_num)])

        if activation_type == 'ReLu':
            self.relus = nn.ModuleList(
                [nn.ReLU(inplace=True) for i in range(local_h_num*local_w_num)])
        elif activation_type == 'PReLu':
            self.relus = nn.ModuleList([nn.PReLU()
                                        for i in range(local_h_num*local_w_num)])

        self.convs = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                              bias) for i in range(local_h_num * local_w_num)])

    def forward(self, x):
        """
        To forward our model, we need to first divide our tensor (picture) into patches.
        Then we use the sets of patch layers to process each parcel.
        """
        h_splits = torch.split(x, int(x.size(2) / self.local_h_num), dim=2)
        h_out = []

        for i in range(len(h_splits)):
            start = True  # Just to see if it is the first layer in the patches
            w_splits = torch.split(h_splits[i], int(
                h_splits[i].size(3)/self.local_w_num), dim=3)
            for j in range(len(w_splits)):
                bn_out = self.bns[i*len(w_splits)+j](w_splits[j].contiguous())
                bn_out = self.relus[i*len(w_splits)+j](bn_out)
                conv_out = self.convs[i*len(w_splits)+j](bn_out)
                if start:
                    h_out.append(conv_out)
                    start = False
                else:
                    h_out[i] = torch.cat((h_out[i], conv_out), 3)
            if i == 0:
                out = h_out[i]
            else:
                out = torch.cat((out, h_out[i]), 2)
        return(out)


class HierarchicalMultiScaleRegionLayer(nn.Module):
    """
    Employs the modules from LocalConv2dReLU to perform patching on our dataset
    """

    def __init__(self, local_group, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, activation_type='ReLU'):
        super(HierarchicalMultiScaleRegionLayer, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, groups, bias)

        self.local_conv_branch1 = LocalConv2dReLU(local_group[0][0], local_group[0][1], out_channels, int(out_channels / 2),
                                                  kernel_size, stride,
                                                  padding, dilation, groups, bias, activation_type)
        self.local_conv_branch2 = LocalConv2dReLU(local_group[1][0], local_group[1][1], int(out_channels / 2),
                                                  int(out_channels /
                                                      4), kernel_size, stride,
                                                  padding, dilation, groups, bias, activation_type)
        self.local_conv_branch3 = LocalConv2dReLU(local_group[2][0], local_group[2][1], int(out_channels / 4),
                                                  int(out_channels /
                                                      4), kernel_size, stride,
                                                  padding, dilation, groups, bias, activation_type)

        self.bn = nn.BatchNorm2d(out_channels) # Not sure why the feature shape is out_channels.

        if activation_type == 'ReLU':
            self.relu = nn.ReLU(inplace=True)
        elif activation_type == 'PReLU':
            self.relu = nn.PReLU()

    def forward(self,x):
        x = self.conv(x)
        local_branch1 = self.local_conv_branch1(x)
        local_branch2 = self.local_conv_branch2(local_branch1)
        local_branch3 = self.local_conv_branch3(local_branch2)
        local_out = torch.cat((local_branch1,local_branch2,local_branch3),1)

        out = x + local_out
        out = self.bn(out)
        out = self.relu(out)

        return(out)



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
        return (multiscale_feat)


def generate_map(map, crop_size, map_size, spatial_ratio, fill_coeff, center1_x, center1_y, center2_x, center2_y):
    """
    For Action units, generate attention map regions with landmarks
    """
    
    spatial_scale = float(map_size)/crop_size #Spatial Scaling?

    half_AU_size = round((map_size - 1) / 2.0 * spatial_ratio) #Some sort of scaling?

    centers = np.array([[center1_x, center1_y],
                       [center2_x, center2_y]]) # Locations for landmarks.

    for center_ind in range(centers.shape[0]):
        AU_center_x = round(centers[center_ind,0] * spatial_scale) # Apply scaling for attention 
        AU_center_y = round(centers[center_ind,1] * spatial_scale) # 
        start_w = round(AU_center_x - half_AU_size) 
        start_h = round(AU_center_y - half_AU_size)
        end_w = round(AU_center_x + half_AU_size)
        end_h = round(AU_center_y + half_AU_size)

        # treat landmark coordinates as starting from 0 rather than 1
        start_h = max(start_h, 0)
        start_h = min(start_h, map_size - 1)
        start_w = max(start_w, 0)
        start_w = min(start_w, map_size - 1)
        end_h = max(end_h, 0)
        end_h = min(end_h, map_size - 1)
        end_w = max(end_w, 0)
        end_w = min(end_w, map_size - 1)

        for h in range(int(start_h), int(end_h)+1):
            for w in range(int(start_w), int(end_w)+1):
                map[h,w] = max(1 - (abs(h - AU_center_y) + abs(w - AU_center_x)) *
                                 fill_coeff / (map_size*spatial_ratio), map[h,w])

class AlignNet(nn.Module):
    def __init__(self, crop_size, map_size, au_num, land_num, input_dim, unit_dim=8,
                 spatial_ratio=0.14, fill_coeff=0.56):
        """
        land_num: number of landmarks (68?)
        au_num: number of Action units (12?)
        map_size: size of attention maps
        unit_dim: not sure what it is?

        """
        super(AlignNet, self).__init__()

        self.align_feat = HLFeatExtractor(input_dim=input_dim, unit_dim=unit_dim)
        self.align_output = nn.Sequential(
            nn.Linear(4000, unit_dim * 64),
            nn.Linear(unit_dim * 64, land_num * 2) # Output landmarks, in terms of x and y coordinates (that's why it has a multiplier of 2)
        )
        self.crop_size = crop_size
        self.map_size = map_size
        self.au_num = au_num
        self.land_num = land_num
        self.spatial_ratio = spatial_ratio
        self.fill_coeff = fill_coeff

    def forward(self, x):
        align_feat_out = self.align_feat(x) #Derived from HLFeatExtractor layers
        align_feat = align_feat_out.view(align_feat_out.size(0), -1) # Convert to 2d array (to be used later to predicts landmarks?)
        align_output = self.align_output(align_feat) # Predicting landmarks. Has shape (unit_dim * 64, land_num * 2)

        aus_map = torch.zeros((align_output.size(0), self.au_num, self.map_size + 8, self.map_size + 8)) # Not sure why they add 8. But is an AU map?

        for i in range(align_output.size(0)):
            land_array = align_output[i,:] # A series of landmarks 
            land_array = land_array.data.cpu().numpy() 
            str_dt = np.append(land_array[0:len(land_array):2], land_array[1:len(land_array):2]) # Split landmark files to x, y coordinates
            arr2d = np.array(str_dt).reshape((2, self.land_num)) # Anothe reshape...
            ruler = abs(arr2d[0, 39] - arr2d[0, 42]) # This is essentially ocular distance

            # See https://www.researchgate.net/figure/Set-of-49-facial-landmarks-of-the-IntraFace-tracker_fig1_314521910 for more details
            #au1
            generate_map(aus_map[i,0], self.crop_size, self.map_size+8, self.spatial_ratio, self.fill_coeff,
                               arr2d[0,21], arr2d[1,21]-ruler/2, arr2d[0,22], arr2d[1,22]-ruler/2)
            # Innermost eyebowns

            # au2
            generate_map(aus_map[i,1], self.crop_size, self.map_size+8, self.spatial_ratio, self.fill_coeff,
                               arr2d[0,18],arr2d[1,18]-ruler/3,arr2d[0,25],arr2d[1,25]-ruler/3)
            # Outmost eyebowns

            # au4
            generate_map(aus_map[i,2], self.crop_size, self.map_size+8, self.spatial_ratio, self.fill_coeff,
                               arr2d[0,19],arr2d[1,19]+ruler/3,arr2d[0,24],arr2d[1,24]+ruler/3)
            # Middle eyebowns

            # au6
            generate_map(aus_map[i,3], self.crop_size, self.map_size+8, self.spatial_ratio, self.fill_coeff,
                               arr2d[0,41],arr2d[1,41]+ruler,arr2d[0,46],arr2d[1,46]+ruler)

            # Someone in the lower eyelid

            # for bp4d
            if self.au_num == 12:

                # au7
                generate_map(aus_map[i,4], self.crop_size, self.map_size+8, self.spatial_ratio, self.fill_coeff,
                                   arr2d[0,38],arr2d[1,38],arr2d[0,43],arr2d[1,43])

                # au10
                generate_map(aus_map[i,5], self.crop_size, self.map_size+8, self.spatial_ratio, self.fill_coeff,
                                   arr2d[0,61],arr2d[1,61],arr2d[0,63],arr2d[1,63])

                # au12 au14 au15
                generate_map(aus_map[i,6], self.crop_size, self.map_size+8, self.spatial_ratio, self.fill_coeff,
                                   arr2d[0,48],arr2d[1,48],arr2d[0,54],arr2d[1,54])
                aus_map[i,7] =aus_map[i,6]
                aus_map[i,8] =aus_map[i,6]

                # au17
                generate_map(aus_map[i,9], self.crop_size, self.map_size+8, self.spatial_ratio, self.fill_coeff,
                                   arr2d[0,56],arr2d[1,56]+ruler/2,arr2d[0,58],arr2d[1,58]+ruler/2)

                # au23 au24
                generate_map(aus_map[i,10], self.crop_size, self.map_size+8, self.spatial_ratio, self.fill_coeff,
                                   arr2d[0,51],arr2d[1,51],arr2d[0,57],arr2d[1,57])
                aus_map[i, 11] = aus_map[i,10]


            # # for disfa
            # elif self.au_num == 8:
            #     # au9
            #     generate_map(aus_map[i, 4], self.crop_size, self.map_size + 8, self.spatial_ratio, self.fill_coeff,
            #                  arr2d[0,15],arr2d[1,15]-ruler/2,arr2d[0,17],arr2d[1,17]-ruler/2)
            #     # au12
            #     generate_map(aus_map[i, 5], self.crop_size, self.map_size + 8, self.spatial_ratio, self.fill_coeff,
            #                  arr2d[0, 31], arr2d[1, 31], arr2d[0, 37], arr2d[1, 37])
            #     # au25
            #     generate_map(aus_map[i, 6], self.crop_size, self.map_size + 8, self.spatial_ratio, self.fill_coeff,
            #                  arr2d[0, 34], arr2d[1, 34], arr2d[0, 40], arr2d[1, 40])
            #     # au26
            #     generate_map(aus_map[i, 7], self.crop_size, self.map_size + 8, self.spatial_ratio, self.fill_coeff,
            #                  arr2d[0, 39], arr2d[1, 39] + ruler / 2, arr2d[0, 41], arr2d[1, 41] + ruler / 2)
            # # for gft
            # elif self.au_num == 10:
            #     # au10
            #     generate_map(aus_map[i, 4], self.crop_size, self.map_size + 8, self.spatial_ratio, self.fill_coeff,
            #                  arr2d[0, 43], arr2d[1, 43], arr2d[0, 45], arr2d[1, 45])
            #     # au12 au14 au15
            #     generate_map(aus_map[i, 5], self.crop_size, self.map_size + 8, self.spatial_ratio, self.fill_coeff,
            #                  arr2d[0, 31], arr2d[1, 31], arr2d[0, 37], arr2d[1, 37])
            #     aus_map[i, 6] = aus_map[i, 5]
            #     aus_map[i, 7] = aus_map[i, 5]
            #     # au23 au24
            #     generate_map(aus_map[i, 8], self.crop_size, self.map_size + 8, self.spatial_ratio, self.fill_coeff,
            #                  arr2d[0, 34], arr2d[1, 34], arr2d[0, 40], arr2d[1, 40])
            #     aus_map[i, 9] = aus_map[i, 8]

        return align_feat_out, align_output, aus_map

class LocalAttentionRefine(nn.Module):

    def __init__(self, au_num, unit_dim=8):

        super(LocalAttentionRefine, self).__init__()
        self.local_aus_attention = nn.ModuleList([nn.Sequential(
            nn.Conv2d(1, unit_dim*8, kernel_size=3, stride = 1, bias=True),
            nn.BatchNorm2d(unit_dim*8),
            nn.ReLU(inplace=True),

            nn.Conv2d(unit_dim*8,unit_dim*8, kernel_size=3, stride=1, bias=True),
            nn.BatchNorm2d(unit_dim*8),
            nn.ReLU(inplace=True),

            nn.Conv2d(unit_dim*8, unit_dim*8, kernel_size=3, stride=1, bias=True),
            nn.BatchNorm2d(unit_dim*8),
            nn.ReLU(unit_dim*8),

            nn.Conv2d(unit_dim*8,1,kernel_size=3,stride=1,bias=True),
            nn.Sigmoid()
        ) for i in range(au_num)])

    def forward(self,x):
        # Assumes that the input x comes directly from AlignNet
        for i in range(len(self.local_aus_attention)):
            initial_au_map = x[:,i,:,:]
            initial_au_map = initial_au_map.unsqueeze(1) # Expand a new dimension
            au_map = self.local_aus_attention[i](initial_au_map)

            if i==0:
                aus_map = au_map
            else:
                aus_map = torch.cat((aus_map,au_map),1)
        return(aus_map)


# class LocalAUNetv1(nn.Module):
#     def __init__(self, au_num, input_dim, unit_dim=8):
#         super(LocalAUNetv1, self).__init__()

#         self.local_aus_branch = nn.ModuleList(
#             [HLFeatExtractor(input_dim=input_dim, unit_dim=unit_dim) for i in range(au_num)])

#     def forward(self, feat, aus_map):
#         for i in range(len(self.local_aus_branch)):
#             au_map = aus_map[:,i,:,:]
#             au_map = au_map.unsqueeze(1)
#             au_feat = feat * au_map
#             output_au_feat = self.local_aus_branch[i](au_feat)
#             if i == 0:
#                 aus_feat = output_au_feat
#             else:
#                 aus_feat = aus_feat + output_au_feat
#         # average over all AUs
#         aus_feat = aus_feat /float(len(self.local_aus_branch))
#         return aus_feat

class LocalAUNetv2(nn.Module):
    def __init__(self, au_num, input_dim, unit_dim=8):
        super(LocalAUNetv2, self).__init__()
        
        self.local_aus_branch = nn.ModuleList(
            [HLFeatExtractor(input_dim=input_dim, unit_dim=unit_dim) for i in range(au_num)]
        )
        self.local_aus_output = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(4000,unit_dim*8),
                nn.Linear(unit_dim*8, 2)
            ) for i in range(au_num)])
        
        def forward(self, feat, aus_map):
            for i in range(len(self.local_aus_output)):
                au_map = aus_map[:,i,:,:]
                au_map = au_map.unsqueeze(1)
                au_feat = feat*au_map
                output_au_feat = self.local_aus_branch[i](au_feat)
                reshape_output_au_feat = output_au_feat.view(output_au_feat.size(0), -1)
                au_output = self.local_aus_output[i](reshape_output_au_feat)
                au_output = au_output.view(au_output.size(0), 2, int(au_output.size(1) / 2))
                au_output = F.log_softmax(au_output, dim=1)

                if i == 0:
                    aus_feat = output_au_feat
                    aus_output = au_output
                else:
                    aus_feat = aus_feat + output_au_feat
                    aus_output = torch.cat((aus_output, au_output), 2)

            # average over all AUs
            aus_feat = aus_feat / float(len(self.local_aus_branch))
            return aus_feat, aus_output

class AUNet(nn.Module):
    def __init__(self, au_num, input_dim = 12000, unit_dim = 8):
        super(AUNet, self).__init__()

        self.au_output = nn.Sequential(
            nn.Linear(input_dim, unit_dim * 64),
            nn.Linear(unit_dim * 64, au_num * 2)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        au_output = self.au_output(x)
        au_output = au_output.view(au_output.size(0), 2, int(au_output.size(1)/2))
        au_output = F.log_softmax(au_output, dim=1)
        return au_output


network_dict = {'HLFeatExtractor':HLFeatExtractor, 'HMRegionLearning':HMRegionLearning,
                'AlignNet':AlignNet, 'LocalAttentionRefine':LocalAttentionRefine,
                'LocalAUNetv2':LocalAUNetv2, 'AUNet':AUNet
                }

