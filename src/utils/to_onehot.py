#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np

def to_onehot(matrix, labels=[], single_foregound_lable=True, background_channel=True, onehot_type=np.dtype(np.float32)):
	matrix = np.around(matrix)
	if len(labels) == 0:
		labels = np.unique(matrix)
		labels = labels[1::]

	mask = np.zeros(matrix.shape, dtype=onehot_type)
	for i, label in enumerate(labels):
		mask += ((matrix == label) * (i+1))

	if single_foregound_lable:
		mask = (mask > 0)
		labels = [1]

	labels_len = len(labels)

	onehot = np.zeros((labels_len+1,) + matrix.shape, dtype=onehot_type)
	for i in range(mask.max()+1):
		onehot[i] = (mask == i)

	if background_channel == False:
		onehot = onehot[1::]

	return mask, onehot, labels

def corrige_label(label):
	if label.max()==8 or label.max()==520:
		label[label == 7] = 1
		label[label == 8] = 2
		label[label == 4] = 3
		label[label == 5] = 4
		label[label == 6] = 5

		label[label > 6] = 0

	return label

def mask_to_onehot(mask):
	#print('Shape:', mask.shape)
	#print('MinMax:', mask.min(), mask.max())

	if len(mask.shape)==5:
		mask = mask.squeeze()

	if len(mask.shape)==4 and mask.shape[0]==6:
		return mask
	else:
		mask = mask.squeeze()

	if torch.is_tensor(mask):
		mask = mask.numpy()

	if len(mask.shape)==3:
		mask = corrige_label(mask)

		if mask.max()==8 or mask.max()==520:
			mask_one, onehot, labels_one = to_onehot(mask, [7,8,4,5,6], single_foregound_lable=False, onehot_type=np.dtype(np.int8))
		elif mask.max()==5:
			mask_one, onehot, labels_one = to_onehot(mask, [1,2,3,4,5], single_foregound_lable=False, onehot_type=np.dtype(np.int8))
		elif mask.max()==6:
			mask_one, onehot, labels_one = to_onehot(mask, [1,2,3,4,5,6], single_foregound_lable=False, onehot_type=np.dtype(np.int8))
		elif mask.max()==7:
			mask_one, onehot, labels_one = to_onehot(mask, [1,2,3,4,5,6,7], single_foregound_lable=False, onehot_type=np.dtype(np.int8))
		elif mask.max()==11:
			mask_one, onehot, labels_one = to_onehot(mask, [1,2,3,4,5,6,7,8,9,10,11], single_foregound_lable=False, onehot_type=np.dtype(np.int8))
		mask = onehot
	else:
		print('Shape errado')

	return mask
