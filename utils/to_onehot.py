#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
