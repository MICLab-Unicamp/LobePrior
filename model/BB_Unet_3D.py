#!/usr/bin/env python
# -*- coding: utf-8 -*-
# https://github.com/rosanajurdi/BB-UNet_UNet_with_bounding_box_prior/tree/master
# https://hal-normandie-univ.archives-ouvertes.fr/hal-02863197

'''
Created on Mar 18, 2020

@author: eljurros
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import Module

class DownConv(Module):
	def __init__(self, in_feat, out_feat, drop_rate=0.4, bn_momentum=0.1):
		super(DownConv, self).__init__()
		self.conv1 = nn.Conv3d(in_feat, out_feat, kernel_size=3, padding=1)
		self.conv1_bn = nn.BatchNorm3d(out_feat, momentum=bn_momentum)
		self.conv1_drop = nn.Dropout3d(drop_rate)

		self.conv2 = nn.Conv3d(out_feat, out_feat, kernel_size=3, padding=1)
		self.conv2_bn = nn.BatchNorm3d(out_feat, momentum=bn_momentum)
		self.conv2_drop = nn.Dropout3d(drop_rate)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = self.conv1_bn(x)
		x = self.conv1_drop(x)

		x = F.relu(self.conv2(x))
		x = self.conv2_bn(x)
		x = self.conv2_drop(x)
		return x


class UpConv(Module):
	def __init__(self, in_feat, out_feat, drop_rate=0.4, bn_momentum=0.1):
		super(UpConv, self).__init__()
		self.up1 = nn.Upsample(scale_factor=2, mode='trilinear')
		self.downconv = DownConv(in_feat, out_feat, drop_rate, bn_momentum)

	def forward(self, x, y):
		x = self.up1(x)
		x = torch.cat([x, y], dim=1)
		x = self.downconv(x)
		return x

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


class Unet(Module):
	"""A reference U-Net model.

	.. seealso::
		Ronneberger, O., et al (2015). U-Net: Convolutional
		Networks for Biomedical Image Segmentation
		ArXiv link: https://arxiv.org/abs/1505.04597
	"""
	def __init__(self,in_dim = 1,  drop_rate=0.4, bn_momentum=0.1, n_organs = 1):
		super(Unet, self).__init__()

		#Downsampling path
		self.conv1 = DownConv(in_dim, 64, drop_rate, bn_momentum)
		self.mp1 = nn.MaxPool3d(2)

		self.conv2 = DownConv(64, 128, drop_rate, bn_momentum)
		self.mp2 = nn.MaxPool3d(2)

		self.conv3 = DownConv(128, 256, drop_rate, bn_momentum)
		self.mp3 = nn.MaxPool3d(2)

		# Bottle neck
		self.conv4 = DownConv(256, 256, drop_rate, bn_momentum)

		# Upsampling path
		self.up1 = UpConv(512, 256, drop_rate, bn_momentum)
		self.up2 = UpConv(384, 128, drop_rate, bn_momentum)
		self.up3 = UpConv(192, 64, drop_rate, bn_momentum)

		self.conv9 = nn.Conv3d(64, n_organs, kernel_size=3, padding=1)

	def forward(self, x, comment = ' '):
		x1 = self.conv1(x)
		p1 = self.mp1(x1)

		x2 = self.conv2(p1)
		p2 = self.mp2(x2)

		x3 = self.conv3(p2)
		p3 = self.mp3(x3)

		# Bottom
		x4 = self.conv4(p3)

		# Up-sampling
		u1 = self.up1(x4, x3)
		u2 = self.up2(u1, x2)
		u3 = self.up3(u2, x1)

		x5 = self.conv9(u3)

		return x5


class BB_Unet(Module):
	"""A reference U-Net model.
	.. seealso::
		Ronneberger, O., et al (2015). U-Net: Convolutional
		Networks for Biomedical Image Segmentation
		ArXiv link: https://arxiv.org/abs/1505.04597
	"""
	def __init__(self, drop_rate=0.6, bn_momentum=0.1, no_grad=False, n_channels=1, n_organs = 6, BB_boxes = 1):
		super(BB_Unet, self).__init__()
		if no_grad is True:
			no_grad_state = True
		else:
			no_grad_state = False

		#Downsampling path
		self.conv1 = DownConv(n_channels, 64, drop_rate, bn_momentum)
		self.mp1 = nn.MaxPool3d(2)

		self.conv2 = DownConv(64, 128, drop_rate, bn_momentum)
		self.mp2 = nn.MaxPool3d(2)

		self.conv3 = DownConv(128, 256, drop_rate, bn_momentum)
		self.mp3 = nn.MaxPool3d(2)

		# Bottle neck
		self.conv4 = DownConv(256, 256, drop_rate, bn_momentum)
		# bounding box encoder path:
		self.b1 = BBConv(BB_boxes, 256, 4, no_grad_state)
		self.b2 = BBConv(BB_boxes, 128, 2, no_grad_state)
		self.b3 = BBConv(BB_boxes, 64, 1, no_grad_state)
		# Upsampling path
		self.up1 = UpConv(512, 256, drop_rate, bn_momentum)
		self.up2 = UpConv(384, 128, drop_rate, bn_momentum)
		self.up3 = UpConv(192, 64, drop_rate, bn_momentum)

		self.conv9 = nn.Conv3d(64, n_organs, kernel_size=3, padding=1)

	def forward(self, data, comment = 'train'):
		#x = data[:,0:7]
		#bb = data[:,7:12]
		x = data[:,0:2]
		bb = data[:,2:3]
		print('Shape', x.shape, bb.shape)

		#x = Variable(torch.randn(1, 3, 32, 64, 64)).cuda()
		#bb = Variable(torch.randn(1, 6, 32, 64, 64)).cuda()

		#self.b1.conv1.requires_grad = False
		#self.b2.conv1.requires_grad = False
		#self.b3.conv1.requires_grad = False
		x1 = self.conv1(x)
		p1 = self.mp1(x1)

		x2 = self.conv2(p1)
		p2 = self.mp2(x2)

		x3 = self.conv3(p2)
		p3 = self.mp3(x3)

		# Bottle neck
		x4 = self.conv4(p3)
		# bbox encoder
		if comment == 'train':
			f1_1 = self.b1(bb)
			f2_1 = self.b2(bb)
			f3_1 = self.b3(bb)
			x3_1 = x3*f1_1
			x2_1 = x2*f2_1
			x1_1 = x1*f3_1
		else:
			x3_1 = x3
			x2_1 = x2
			x1_1 = x1

		# Up-sampling
		u1 = self.up1(x4, x3_1)
		u2 = self.up2(u1, x2_1)
		u3 = self.up3(u2, x1_1)
		x5 = self.conv9(u3)
		return x5

if __name__ == '__main__':
	import os
	os.system('cls' if os.name == 'nt' else 'clear')

	import torch
	from torch.autograd import Variable

	#torch.cuda.set_device(0)
	net = BB_Unet(n_channels=7, n_organs = 5, BB_boxes=5).cuda().eval()

	data = Variable(torch.randn(1, 12, 32, 64, 64)).cuda()
	print(data.shape)

	out = net(data, comment = 'train')
	print("out size: {}".format(out.size()))

	out = net(data)
	print("out size: {}".format(out.size()))





