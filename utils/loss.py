#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import L1Loss
from monai.losses import DiceLoss, GeneralizedDiceLoss

class FocalLossKaggle(nn.Module):
	def __init__(self, weight=None, size_average=True):
		super(FocalLossKaggle, self).__init__()

	def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):
		assert inputs.shape==targets.shape

		#comment out if your model contains a sigmoid or equivalent activation layer
		inputs = torch.sigmoid(inputs)

		#flatten label and prediction tensors
		#inputs = inputs.view(-1)
		#targets = targets.view(-1)
		inputs = inputs.reshape(-1)
		targets = targets.reshape(-1)

		#first compute binary cross-entropy
		BCE = F.binary_cross_entropy(inputs.float(), targets.float(), reduction='mean')
		BCE_EXP = torch.exp(-BCE)
		focal_loss = alpha * (1-BCE_EXP)**gamma * BCE

		return focal_loss

class DiceLoss_chavg(nn.Module):

	def __init__(self):
		super(DiceLoss_chavg, self).__init__()
		self.smooth = 1e-5

	def forward(self, y_pred, y_true):
		assert y_pred.size() == y_true.size()
		#if not (y_pred.size() == y_true.size()):
		#    raise ValueError("Target size ({}) must be the same as input size ({})".format(y_pred.size(), y_true.size()))
		#_,nch,_,_ = y_true.size()
		nch = y_true.shape[1]
		dsc = 0
		for i in range(nch):
			y_pred_aux = y_pred[:, i].contiguous().view(-1)
			y_true_aux = y_true[:, i].contiguous().view(-1)
			intersection = (y_pred_aux * y_true_aux).sum()
			dsc += (2. * intersection + self.smooth) / (
				y_pred_aux.sum() + y_true_aux.sum() + self.smooth
			)

		return 1. - dsc/(nch)

class DiceLoss_weighs(nn.Module):
	#def __str__(self):
	#    return f"DiceLoss_weighs({self.weights})"

	# 0,025 + 0,2 + 0,175 + 0,2 + 0,2 + 0,2 = 0.025, 0.2, 0.175, 0.2, 0.2, 0.2
	def __init__(self, weights=[0.025, 0.2, 0.175, 0.2, 0.2, 0.2]):
		super(DiceLoss_weighs, self).__init__()
		assert (np.sum(weights)==1)
		self.smooth = 1.0
		self.weights = weights

	def forward(self, y_pred, y_true, pow=False):
		assert y_pred.size() == y_true.size()
		nch = y_true.shape[1]

		dsc = 0
		for i in range(nch):
			y_pred_aux = y_pred[:, i].contiguous().view(-1)
			y_true_aux = y_true[:, i].contiguous().view(-1)
			intersection = (y_pred_aux * y_true_aux).sum()
			if pow:
				dsc += (2. * intersection + self.smooth) / (
					y_pred_aux.sum()**2 + y_true_aux.sum()**2 + self.smooth
				) * self.weights[i]
			else:
				dsc += (2. * intersection + self.smooth) / (
					y_pred_aux.sum() + y_true_aux.sum() + self.smooth
				) * self.weights[i]

		return 1. - dsc

class CombinedLoss(nn.Module):
    def __init__(self, include_background, cross_entropy=False, gdl=False, soft_circulatory=False):
        super().__init__()
        self.include_background = include_background
        self.gdl = gdl
        self.cross_entropy = cross_entropy
        self.soft_circulatory = soft_circulatory

        if self.gdl:
            dice_str = "MONAI GDL"
            self.dice = GeneralizedDiceLoss(include_background=self.include_background)
        else:
            dice_str = "MONAI DiceLoss"
            self.dice = DiceLoss(include_background=self.include_background)

        if self.cross_entropy:
            self.cross_entropy_loss = nn.NLLLoss()
            self.initialization_string = f"CombinedLoss combining {dice_str} and torch NLLLoss, include background: {self.include_background}"
            print(f"WARNING: Cross Entropy always is including background! {self.include_background} is only affecting {dice_str}")
        else:
            self.l1loss = L1Loss()
            self.initialization_string = f"CombinedLoss combining {dice_str} and torch L1Loss, include background: {self.include_background}"

        print(self.initialization_string)

class BayesianLoss(torch.nn.Module):
  def __init__(self, num_classes=6):
    super(BayesianLoss, self).__init__()
    self.num_classes = num_classes

  def forward(self, y_pred, y_true, atlas):
    # Compute the bayesian cross-entropy between the true labels and the predicted labels.
    loss = torch.nn.functional.cross_entropy(y_pred, y_true)
    loss = FocalLossKaggle()(y_pred, y_true)

    # Compute the posterior probabilities for each class.
    #posterior_probabilities = torch.nn.functional.softmax(y_pred, dim=1)

    # Compute the log posterior probabilities for each class.
    log_posterior_probabilities = torch.nn.functional.log_softmax(y_pred, dim=1)

    # Compute the prior probabilities for each class.
    #prior_probabilities = self.model.get_prior_probabilities()
    #prior_probabilities = torch.nn.functional.softmax(atlas, dim=1)

    # Compute the log prior probabilities for each class.
    log_prior_probabilities = torch.nn.functional.log_softmax(atlas, dim=1)

    # Compute the bayesian loss for each example.
    bayesian_loss = loss + torch.sum(log_posterior_probabilities - log_prior_probabilities, dim=1).mean()

    return 1-bayesian_loss

class BayesianLossByChannel(torch.nn.Module):
	def __init__(self, num_classes=6):
		super(BayesianLossByChannel, self).__init__()
		self.num_classes = num_classes

	def forward(self, y_pred, y_true, atlas):
		# Compute the bayesian cross-entropy between the true labels and the predicted labels.
		#loss = torch.nn.functional.cross_entropy(y_pred, y_true)
		loss = FocalLossKaggle()(y_pred, y_true)

		# Compute the posterior probabilities for each class.
		#posterior_probabilities = torch.nn.functional.softmax(y_pred, dim=1)

		# Compute the log posterior probabilities for each class.
		#log_posterior_probabilities = torch.nn.functional.log_softmax(y_pred, dim=1)

		# Compute the prior probabilities for each class.
		#prior_probabilities = self.model.get_prior_probabilities()
		#prior_probabilities = torch.nn.functional.softmax(atlas, dim=1)

		# Compute the log prior probabilities for each class.
		#log_prior_probabilities = torch.nn.functional.log_softmax(atlas, dim=1)

		# Compute the bayesian loss for each example.
		#bayesian_loss = loss + torch.sum(log_posterior_probabilities - log_prior_probabilities, dim=1).mean()
		bayesian_loss = loss + torch.sum(y_pred - atlas, dim=1).mean()

		return bayesian_loss

def main(args):
	image = Variable(torch.randn(1, 6, 64, 256, 256))
	label = Variable(torch.randn(1, 6, 64, 256, 256))
	data = torch.randn(1, 6, 64, 256, 256)

	#data = np.random.randint(0.1,1.0, (6,300,256,256))
	#data = np.zeros((1,6,64,256,256))
	#data = np.ones((1,6,64,256,256))

	focal_loss = FocalLossKaggle()(image,label)
	print(f'Final loss: {focal_loss}')

	dice_loss = DiceLoss_chavg()(image,label)
	print(f'Final loss: {dice_loss}')

	dice_loss_weighs = DiceLoss_weighs()(image,label)
	print(f'Final loss: {dice_loss_weighs}')

	bayesian_loss = BayesianLoss()(image,label,label)
	print(f'Final loss: {bayesian_loss}')

	return 0

if __name__ == '__main__':
	os.system('cls' if os.name == 'nt' else 'clear')
	sys.exit(main(sys.argv))
