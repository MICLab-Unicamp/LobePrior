#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module

# Ref https://github.com/Thvnvtos/Lung_Segmentation

# __                            __
#  1|__   ________________   __|1
#     2|__  ____________  __|2
#        3|__  ______  __|3
#           4|__ __ __|4

class ConvUnit(nn.Module):
	"""
		Convolution Unit: (Conv3D -> BatchNorm -> ReLu) * 2
	"""
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.double_conv = nn.Sequential(
			nn.Conv3d(in_channels, out_channels, kernel_size = 3, padding = 1),
			nn.InstanceNorm3d(out_channels),
			nn.ReLU(inplace=True), # inplace=True means it changes the input directly, input is lost

			nn.Conv3d(out_channels, out_channels, kernel_size = 3, padding = 1),
			nn.InstanceNorm3d(out_channels),
			nn.ReLU(inplace=True)
		  )

	def forward(self,x):
		return self.double_conv(x)

class EncoderUnit(nn.Module):
	"""
	An Encoder Unit with the ConvUnit and MaxPool
	"""
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.encoder = nn.Sequential(
			nn.MaxPool3d(2),
			ConvUnit(in_channels, out_channels)
		)
	def forward(self, x):
		return self.encoder(x)

class DecoderUnit(nn.Module):
	"""
	ConvUnit and upsample with Upsample or convTranspose
	"""
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
		self.conv = ConvUnit(in_channels, out_channels)

	def forward(self, x1, x2):
		x1 = self.up(x1)

		diffZ = x2.size()[2] - x1.size()[2]
		diffY = x2.size()[3] - x1.size()[3]
		diffX = x2.size()[4] - x1.size()[4]
		x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])

		x = torch.cat([x2, x1], dim=1)
		return self.conv(x)

class OutConv(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.conv = nn.Conv3d(in_channels, out_channels, kernel_size = 1)

	def forward(self, x):
		return self.conv(x)

class BBConv(Module):
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
		x = F.sigmoid(x)
		return x

class UNet3d(nn.Module):
	def __init__(self, in_channels, n_classes, s_channels):
		super().__init__()
		self.in_channels = in_channels
		self.n_classes = n_classes
		self.s_channels = s_channels

		self.conv = ConvUnit(in_channels, s_channels)
		self.enc1 = EncoderUnit(s_channels, 2 * s_channels)
		self.enc2 = EncoderUnit(2 * s_channels, 4 * s_channels)
		self.enc3 = EncoderUnit(4 * s_channels, 8 * s_channels)
		self.enc4 = EncoderUnit(8 * s_channels, 8 * s_channels)

		self.dec1 = DecoderUnit(16 * s_channels, 4 * s_channels)
		self.dec2 = DecoderUnit(8 * s_channels, 2 * s_channels)
		self.dec3 = DecoderUnit(4 * s_channels, s_channels)
		self.dec4 = DecoderUnit(2 * s_channels, s_channels)
		self.out = OutConv(s_channels, n_classes)

	def forward(self, x):
		x1 = self.conv(x)
		x2 = self.enc1(x1)
		x3 = self.enc2(x2)
		x4 = self.enc3(x3)
		x5 = self.enc4(x4)

		mask = self.dec1(x5, x4)
		mask = self.dec2(mask, x3)
		mask = self.dec3(mask, x2)
		mask = self.dec4(mask, x1)
		mask = self.out(mask)
		return mask#, x5

class BB_Unet_pytorch(nn.Module):
	def __init__(self, in_channels, n_classes, s_channels, no_grad=False, BB_boxes = 1):
		super().__init__()
		self.in_channels = in_channels
		self.n_classes = n_classes
		self.s_channels = s_channels

		if no_grad is True:
			no_grad_state = True
		else:
			no_grad_state = False

		self.conv = ConvUnit(in_channels, s_channels)
		self.enc1 = EncoderUnit(s_channels, 2 * s_channels)
		self.enc2 = EncoderUnit(2 * s_channels, 4 * s_channels)
		self.enc3 = EncoderUnit(4 * s_channels, 8 * s_channels)
		self.enc4 = EncoderUnit(8 * s_channels, 8 * s_channels)
		
		self.b0 = BBConv(BB_boxes, 512, 8, no_grad_state)
		self.b1 = BBConv(BB_boxes, 256, 4, no_grad_state)
		self.b2 = BBConv(BB_boxes, 128, 2, no_grad_state)
		self.b3 = BBConv(BB_boxes, 64, 1, no_grad_state)

		self.dec1 = DecoderUnit(16 * s_channels, 4 * s_channels)
		self.dec2 = DecoderUnit(8 * s_channels, 2 * s_channels)
		self.dec3 = DecoderUnit(4 * s_channels, s_channels)
		self.dec4 = DecoderUnit(2 * s_channels, s_channels)
		self.out = OutConv(s_channels, n_classes)

	def forward(self, data, comment = 'train'):
		x = data[:,0:1]
		bb = data[:,1:2]
		#print('Shape', x.shape, bb.shape)

		x1 = self.conv(x)
		x2 = self.enc1(x1)
		x3 = self.enc2(x2)
		x4 = self.enc3(x3)
		x5 = self.enc4(x4)
		#print(x1.shape, x2.shape, x3.shape, x4.shape, x5.shape)

		if comment == 'train':
			f0_1 = self.b0(bb)
			f1_1 = self.b1(bb)
			f2_1 = self.b2(bb)
			f3_1 = self.b3(bb)
			#print(f0_1.shape, f1_1.shape, f2_1.shape, f3_1.shape)

			x4_1 = x4*f0_1
			x3_1 = x3*f1_1
			x2_1 = x2*f2_1
			x1_1 = x1*f3_1
		else:
			x4_1 = x4
			x3_1 = x3
			x2_1 = x2
			x1_1 = x1

		mask = self.dec1(x5, x4_1)
		mask = self.dec2(mask, x3_1)
		mask = self.dec3(mask, x2_1)
		mask = self.dec4(mask, x1_1)
		mask = self.out(mask)
		return mask#, x5

if __name__ == '__main__':
	import os
	os.system('cls' if os.name == 'nt' else 'clear')

	import torch
	from torch.autograd import Variable
	from torchsummary import summary

	#torch.cuda.set_device(0)

	net = BB_Unet_pytorch(in_channels=1, n_classes=6, s_channels=64, BB_boxes=1).cuda().eval()

	data = Variable(torch.randn(1, 2, 32, 128, 128)).cuda()
	print(data.shape)

	out = net(data, comment = 'train')
	print("out size: {}".format(out.size()))

	out = net(data)
	print("out size: {}".format(out.size()))
