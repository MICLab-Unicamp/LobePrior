#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
from .to_onehot import mask_to_onehot

class Dice_chavg_per_label_metric(nn.Module):

	def __init__(self):
		super(Dice_chavg_per_label_metric, self).__init__()
		self.smooth = 0  # use as metric

	def forward(self, y_pred, y_true):
		assert y_pred.size() == y_true.size(), f'{y_pred.shape} {y_true.shape}'
		#_,nch,_,_ = y_true.size()
		nch = y_true.shape[1]
		dsc = 0
		dsc_per_label = []
		for i in range(nch):
			y_pred_aux = y_pred[:, i].contiguous().view(-1)
			y_true_aux = y_true[:, i].contiguous().view(-1)
			intersection = (y_pred_aux * y_true_aux).sum()
			dsc += (2. * intersection + self.smooth) / (
				y_pred_aux.sum() + y_true_aux.sum() + self.smooth
			)
			dsc_per_label.append((2. * intersection + self.smooth) / (
				y_pred_aux.sum() + y_true_aux.sum() + self.smooth
			))

		return dsc/(nch), dsc_per_label
