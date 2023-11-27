#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act, Conv, Dropout, Norm, split_args
from monai.utils import deprecated_arg

__all__ = ["VNet"]


def get_acti_layer(act: tuple[str, dict] | str, nchan: int = 0):
    if act == "prelu":
        act = ("prelu", {"num_parameters": nchan})
    act_name, act_args = split_args(act)
    act_type = Act[act_name]
    return act_type(**act_args)


class LUConv(nn.Module):
    def __init__(self, spatial_dims: int, nchan: int, act: tuple[str, dict] | str, bias: bool = False):
        super().__init__()

        self.act_function = get_acti_layer(act, nchan)
        self.conv_block = Convolution(
            spatial_dims=spatial_dims,
            in_channels=nchan,
            out_channels=nchan,
            kernel_size=5,
            act=None,
            norm=Norm.BATCH,
            bias=bias,
        )

    def forward(self, x):
        out = self.conv_block(x)
        out = self.act_function(out)
        return out


def _make_nconv(spatial_dims: int, nchan: int, depth: int, act: tuple[str, dict] | str, bias: bool = False):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(spatial_dims, nchan, act, bias))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(
        self, spatial_dims: int, in_channels: int, out_channels: int, act: tuple[str, dict] | str, bias: bool = False
    ):
        super().__init__()

        if out_channels % in_channels != 0:
            raise ValueError(
                f"out channels should be divisible by in_channels. Got in_channels={in_channels}, out_channels={out_channels}."
            )

        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act_function = get_acti_layer(act, out_channels)
        self.conv_block = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=5,
            act=None,
            norm=Norm.BATCH,
            bias=bias,
        )

    def forward(self, x):
        out = self.conv_block(x)
        repeat_num = self.out_channels // self.in_channels
        x16 = x.repeat([1, repeat_num, 1, 1, 1][: self.spatial_dims + 2])
        out = self.act_function(torch.add(out, x16))
        return out


class DownTransition(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        nconvs: int,
        act: tuple[str, dict] | str,
        dropout_prob: float | None = None,
        dropout_dim: int = 3,
        bias: bool = False,
    ):
        super().__init__()

        conv_type: type[nn.Conv2d | nn.Conv3d] = Conv[Conv.CONV, spatial_dims]
        norm_type: type[nn.BatchNorm2d | nn.BatchNorm3d] = Norm[Norm.BATCH, spatial_dims]
        dropout_type: type[nn.Dropout | nn.Dropout2d | nn.Dropout3d] = Dropout[Dropout.DROPOUT, dropout_dim]

        out_channels = 2 * in_channels
        self.down_conv = conv_type(in_channels, out_channels, kernel_size=2, stride=2, bias=bias)
        self.bn1 = norm_type(out_channels)
        self.act_function1 = get_acti_layer(act, out_channels)
        self.act_function2 = get_acti_layer(act, out_channels)
        self.ops = _make_nconv(spatial_dims, out_channels, nconvs, act, bias)
        self.dropout = dropout_type(dropout_prob) if dropout_prob is not None else None

    def forward(self, x):
        down = self.act_function1(self.bn1(self.down_conv(x)))
        if self.dropout is not None:
            out = self.dropout(down)
        else:
            out = down
        out = self.ops(out)
        out = self.act_function2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        nconvs: int,
        act: tuple[str, dict] | str,
        dropout_prob: tuple[float | None, float] = (None, 0.5),
        dropout_dim: int = 3,
    ):
        super().__init__()

        conv_trans_type: type[nn.ConvTranspose2d | nn.ConvTranspose3d] = Conv[Conv.CONVTRANS, spatial_dims]
        norm_type: type[nn.BatchNorm2d | nn.BatchNorm3d] = Norm[Norm.BATCH, spatial_dims]
        dropout_type: type[nn.Dropout | nn.Dropout2d | nn.Dropout3d] = Dropout[Dropout.DROPOUT, dropout_dim]

        self.up_conv = conv_trans_type(in_channels, out_channels // 2, kernel_size=2, stride=2)
        self.bn1 = norm_type(out_channels // 2)
        self.dropout = dropout_type(dropout_prob[0]) if dropout_prob[0] is not None else None
        self.dropout2 = dropout_type(dropout_prob[1])
        self.act_function1 = get_acti_layer(act, out_channels // 2)
        self.act_function2 = get_acti_layer(act, out_channels)
        self.ops = _make_nconv(spatial_dims, out_channels, nconvs, act)

    def forward(self, x, skipx):
        if self.dropout is not None:
            out = self.dropout(x)
        else:
            out = x
        skipxdo = self.dropout2(skipx)
        out = self.act_function1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.act_function2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):
    def __init__(
        self, spatial_dims: int, in_channels: int, out_channels: int, act: tuple[str, dict] | str, bias: bool = False
    ):
        super().__init__()

        conv_type: type[nn.Conv2d | nn.Conv3d] = Conv[Conv.CONV, spatial_dims]

        self.act_function1 = get_acti_layer(act, out_channels)
        self.conv_block = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=5,
            act=None,
            norm=Norm.BATCH,
            bias=bias,
        )
        self.conv2 = conv_type(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # convolve 32 down to 2 channels
        out = self.conv_block(x)
        out = self.act_function1(out)
        out = self.conv2(out)
        return out

class BBConv(nn.Module):
	def __init__(self, in_feat, out_feat, pool_ratio, no_grad_state):
		super(BBConv, self).__init__()
		self.mp = nn.MaxPool3d(pool_ratio)
		self.conv1 = nn.Conv3d(in_feat, out_feat, kernel_size=3, padding=1)
		if no_grad_state is True:
			self.conv1.requires_grad = False
		else:
			self.conv1.requires_grad = True
	def forward(self, x):
		x = self.mp(x)
		x = self.conv1(x)
		x = torch.sigmoid(x)
		return x

class BB_VNet(nn.Module):   
    @deprecated_arg(
        name="dropout_prob",
        since="1.2",
        new_name="dropout_prob_down",
        msg_suffix="please use `dropout_prob_down` instead.",
    )
    @deprecated_arg(
        name="dropout_prob", since="1.2", new_name="dropout_prob_up", msg_suffix="please use `dropout_prob_up` instead."
    )
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 1,
        act: tuple[str, dict] | str = ("elu", {"inplace": True}),
        dropout_prob: float | None = 0.5,  # deprecated
        dropout_prob_down: float | None = 0.5,
        dropout_prob_up: tuple[float | None, float] = (0.5, 0.5),
        dropout_dim: int = 3,
        bias: bool = False,
        no_grad=False,
        BB_boxes = 1,
    ):
        super().__init__()

        if no_grad is True:
            no_grad_state = True
        else:
            no_grad_state = False

        if spatial_dims not in (2, 3):
            raise AssertionError("spatial_dims can only be 2 or 3.")

        self.in_tr = InputTransition(spatial_dims, in_channels, 16, act, bias=bias)
        self.down_tr32 = DownTransition(spatial_dims, 16, 1, act, bias=bias)
        self.down_tr64 = DownTransition(spatial_dims, 32, 2, act, bias=bias)
        self.down_tr128 = DownTransition(spatial_dims, 64, 3, act, dropout_prob=dropout_prob_down, bias=bias)
        self.down_tr256 = DownTransition(spatial_dims, 128, 2, act, dropout_prob=dropout_prob_down, bias=bias)

        # bounding box encoder path:
        self.b1 = BBConv(BB_boxes, 256, 4, no_grad_state)
        self.b2 = BBConv(BB_boxes, 128, 2, no_grad_state)
        self.b3 = BBConv(BB_boxes, 64, 1, no_grad_state)
        self.b4 = BBConv(BB_boxes, 32, 1, no_grad_state)

        self.up_tr256 = UpTransition(spatial_dims, 256, 256, 2, act, dropout_prob=dropout_prob_up)
        self.up_tr128 = UpTransition(spatial_dims, 256, 128, 2, act, dropout_prob=dropout_prob_up)
        self.up_tr64 = UpTransition(spatial_dims, 128, 64, 1, act)
        self.up_tr32 = UpTransition(spatial_dims, 64, 32, 1, act)
        self.out_tr = OutputTransition(spatial_dims, 32, out_channels, act, bias=bias)

    def forward(self, data, comment = 'test'):
        x = data[:,0:1]
        bb = data[:,1:2]

        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        
        if comment == 'train':
            f1_1 = self.b1(bb)
            f2_1 = self.b2(bb)
            f3_1 = self.b3(bb)
            f4_1 = self.b4(bb)
            x4_1 = out128*f1_1
            x3_1 = out64*f2_1
            x2_1 = out32*f3_1
            x1_1 = out16*f4_1
            x = self.up_tr256(out256, x4_1)
            x = self.up_tr128(x, x3_1)
            x = self.up_tr64(x, f2_1)
            x = self.up_tr32(x, x1_1)
        else:
            x4_1 = out128
            x3_1 = out64
            x2_1 = out32
            x1_1 = out16
            x = self.up_tr256(out256, x4_1)
            x = self.up_tr128(x, x3_1)
            x = self.up_tr64(x, x2_1)
            x = self.up_tr32(x, x1_1)

        x = self.out_tr(x)
        return x

if __name__ == '__main__':
	import os
	os.system('cls' if os.name == 'nt' else 'clear')

	import monai
	import torch

	net = BB_VNet(spatial_dims=3, in_channels=1, out_channels=1, dropout_dim=3).cuda().eval()
	test_data = torch.randn(1, 2, 32, 32, 32)
	print("input size: {}".format(output.size()))

	output = net(test_data)
	print("out size: {}".format(output.size()))

	output = net(test_data, comment='test')
	print("out size: {}".format(output.size()))





