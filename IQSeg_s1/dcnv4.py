# --------------------------------------------------------
# Deformable Convolution v4
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
from ..functions import DCNv4Function

class CenterFeatureScaleModule(nn.Module):
    def forward(self,
                query,
                center_feature_scale_proj_weight,
                center_feature_scale_proj_bias):
        center_feature_scale = F.linear(query,
                                        weight=center_feature_scale_proj_weight,
                                        bias=center_feature_scale_proj_bias).sigmoid()
        return center_feature_scale


class DCNv4(nn.Module):
    def __init__(
            self,
            channels=2048,
            kernel_size=3,
            stride=1,
            pad=1,
            dilation=1,
            group=4,
            offset_scale=1.0,
            dw_kernel_size=3,
            center_feature_scale=False,
            remove_center=False,
            output_bias=True,
            without_pointwise=False,
            **kwargs):
        """
        DCNv4 Module
        :param channels
        :param kernel_size
        :param stride
        :param pad
        :param dilation
        :param group
        :param offset_scale
        :param act_layer
        :param norm_layer
        """
        super().__init__()
        if channels % group != 0:
            raise ValueError(
                f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group

        # you'd better set _d_per_group to a power of 2 which is more efficient in our CUDA implementation
        assert _d_per_group % 16 == 0

        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.dw_kernel_size = dw_kernel_size
        self.center_feature_scale = center_feature_scale
        self.remove_center = int(remove_center)
        self.without_pointwise = without_pointwise

        self.K =  group * (kernel_size * kernel_size - self.remove_center)
        if dw_kernel_size is not None:
            self.offset_mask_dw = nn.Conv2d(channels, channels, dw_kernel_size, stride=1, padding=(dw_kernel_size - 1) // 2, groups=channels)
        self.offset_mask = nn.Linear(channels, int(math.ceil((self.K * 3)/8)*8)) #输出的通道数为8的倍数
        if not without_pointwise:
            self.value_proj = nn.Linear(channels, channels)
            self.value_proj2 = nn.Linear(channels, channels)
            self.output_proj = nn.Linear(channels, channels, bias=output_bias)
        self._reset_parameters()

        self.conv1 = nn.Conv2d(channels * 2, channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        if center_feature_scale:
            self.center_feature_scale_proj_weight = nn.Parameter(torch.zeros((group, channels), dtype=torch.float))
            self.center_feature_scale_proj_bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group, ))
            self.center_feature_scale_module = CenterFeatureScaleModule()

    def _reset_parameters(self):
        constant_(self.offset_mask.weight.data, 0.)
        constant_(self.offset_mask.bias.data, 0.)
        if not self.without_pointwise:
            xavier_uniform_(self.value_proj.weight.data)
            constant_(self.value_proj.bias.data, 0.)
            xavier_uniform_(self.output_proj.weight.data)
            if self.output_proj.bias is not None:
                constant_(self.output_proj.bias.data, 0.)

    def forward(self, input1, input2, shape=None):
        """
        :param input1                      (N, C, H, W) 
        :param input2                      (N, C, H, W)
        :return output                     (N, C, H, W)
        """
        N, C, H, W = input1.shape
        x = torch.cat([input1, input2], dim=1)   # NCHW
        x = self.lrelu(self.conv1(x)).permute(0, 2, 3, 1)  # NHWC

        if not self.without_pointwise:
            x = self.value_proj(x)
            x2 = self.value_proj2(input2.permute(0, 2, 3, 1)) # NHWC
        input = x
        #todo offset的维度一般为(B, 2*kernel_size*kernel_size*group, H, W)
        #todo mask的维度一般为(B, kernel_size*kernel_size*group, H, W)
        #todo 正常的操作是输入input经过linear射后经过dw层分别再经过 linear 层得到 offset和 mask，然后cat后pad得到 112维 输出通道
        #todo 这里一步到位，直接通过dw层后，直接输出 3 * kernel_size * kernel_size * group / 8 * 8 之后的112维        
        if self.dw_kernel_size is not None:
            offset_mask_input = self.offset_mask_dw(input.view(N, H, W, C).permute(0, 3, 1, 2)) # NCHW
            offset_mask_input = offset_mask_input.permute(0, 2, 3, 1) # NHWC
            # offset_mask_input = offset_mask_input.permute(0, 2, 3, 1).view(N, L, C)
        else:
            offset_mask_input = input
        offset_mask = self.offset_mask(offset_mask_input)
        # offset_mask = self.offset_mask(offset_mask_input).reshape(N, H, W, -1)

        x_proj = x

        x = DCNv4Function.apply(
            x2, offset_mask,
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.pad, self.pad,
            self.dilation, self.dilation,
            self.group, self.group_channels,
            self.offset_scale,
            256,
            self.remove_center
            )

        if self.center_feature_scale: # 如果需要特征缩放
            center_feature_scale = self.center_feature_scale_module(x, self.center_feature_scale_proj_weight, self.center_feature_scale_proj_bias)
            # x1: (N, H, W, C) -> (N, H, W, groups)
            center_feature_scale = center_feature_scale[..., None].repeat(1, 1, 1, 1, self.channels // self.group).flatten(-2)
            # (N, H, W, groups) -> (N, H, W, groups, 1) -> (N, H, W, groups, _d_per_group) -> (N, H, W, channels)
            x = x * (1 - center_feature_scale) + x_proj * center_feature_scale
            # x_proj和 x 按照一定的系数相加
            
        if not self.without_pointwise:
            x = self.output_proj(x)
        return x.permute(0, 3, 1, 2)  # NCHW