#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn
import numpy as np

def Dice_metric(img, target, treshold=0.5):
    img = img > treshold
    target = target > treshold

    inter = (img * target).sum(-1).sum(-1)*1.0
    soma = (img.sum(-1).sum(-1) + target.sum(-1).sum(-1))*1.0
    soma[soma == 0] = 1e-10

    dc = ((2 * inter) / soma)
    dc[soma == 1e-10] = 1

    return dc

def Jaccard_metric(img, target, treshold=0.5):
    img = img > treshold
    target = target > treshold

    inter = (img * target).sum(-1).sum(-1)*1.0
    soma = (img.sum(-1).sum(-1) + target.sum(-1).sum(-1))*1.0
    soma[soma == 0] = 1e-10

    dc = (inter / soma)
    dc[soma == 1e-10] = 1

    return dc

class Dice_chavg(nn.Module):

    def __init__(self):
        super(Dice_chavg, self).__init__()
        self.smooth = 0  # use as metric

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
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

        return dsc/(nch)

class Dice_chavg_per_label_metric(nn.Module):

    def __init__(self):
        super(Dice_chavg_per_label_metric, self).__init__()
        self.smooth = 0  # use as metric

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
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

class Jaccard_chavg_per_label(nn.Module):

    def __init__(self, smooth=0):
        super(Jaccard_chavg_per_label, self).__init__()
        self.smooth = smooth  # 0 used as metric

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        #_,nch,_,_ = y_true.size()
        nch = y_true.shape[1]
        jaccard = 0
        jaccard_per_label = []
        for i in range(nch):
            y_pred_aux = y_pred[:, i].contiguous().view(-1)
            y_true_aux = y_true[:, i].contiguous().view(-1)
            intersection = (y_pred_aux * y_true_aux).sum()
            jaccard += (intersection + self.smooth) / (
                ((y_pred_aux.sum() + y_true_aux.sum()) - intersection) + self.smooth
            )
            jaccard_per_label.append((intersection + self.smooth) / (
                ((y_pred_aux.sum() + y_true_aux.sum()) - intersection) + self.smooth
            ))

        return jaccard/(nch), jaccard_per_label

# https://github.com/MICLab-Unicamp/tahalmus_benchmark_diffusion_dev/blob/main/code/Utils/Metrics.py
def DiceMetric_weighs_smooth(y_pred, y_true,weights, treshold=None):
    # print('loss y_pred.shape = ', y_pred.shape)
    # print('loss y_true.shape = ', y_true.shape)
    assert y_pred.size() == y_true.size()
    _,nch,_,_ = y_true.size()
    dsc = 0
    smooth = 1e-10

    if treshold:
        y_pred = y_pred > treshold
        y_true = y_true > treshold

    for i in range(nch):
        y_pred_aux = y_pred[:, i].contiguous().view(-1)
        y_true_aux = y_true[:, i].contiguous().view(-1)
        intersection = (y_pred_aux * y_true_aux).sum()
        dsc += (2. * intersection + smooth) / (
            y_pred_aux.sum() + y_true_aux.sum() + smooth
        ) * weights[i]

    return dsc

# https://github.com/MICLab-Unicamp/tahalmus_benchmark_diffusion_dev/blob/main/code/Utils/Metrics.py
def DiceMetric_weighs(y_pred, y_true,weights, treshold=None):
    # print('loss y_pred.shape = ', y_pred.shape)
    # print('loss y_true.shape = ', y_true.shape)
    assert y_pred.size() == y_true.size()
    _,nch,_,_ = y_true.size()
    dsc = 0

    if treshold:
        y_pred = y_pred > treshold
        y_true = y_true > treshold

    for i in range(nch):
        y_pred_aux = y_pred[:, i].contiguous().view(-1)
        y_true_aux = y_true[:, i].contiguous().view(-1)
        if (y_pred_aux.sum() + y_true_aux.sum()) > 0:
            intersection = (y_pred_aux * y_true_aux).sum()
            dsc += (2. * intersection) / (
                y_pred_aux.sum() + y_true_aux.sum()
            ) * weights[i]

    return dsc

# https://github.com/MICLab-Unicamp/tahalmus_benchmark_diffusion_dev/blob/main/code/Utils/Metrics.py
def DiceMetric_weighs_np(y_pred, y_true,weights, treshold=None):
    # print('loss y_pred.shape = ', y_pred.shape)
    # print('loss y_true.shape = ', y_true.shape)
    assert y_pred.shape == y_true.shape
    _,nch,_,_ = y_true.shape
    dsc = 0

    if treshold:
        y_pred = y_pred > treshold
        y_true = y_true > treshold

    for i in range(nch):
        y_pred_aux = y_pred[:, i]
        y_true_aux = y_true[:, i]
        if (y_pred_aux.sum() + y_true_aux.sum()) > 0:
            intersection = (y_pred_aux * y_true_aux).sum()
            dsc += ((2. * intersection) / (y_pred_aux.sum() + y_true_aux.sum())) * weights[i]

    return dsc

# https://github.com/udaykamal20/Team_Spectrum/blob/master/src/get_metrics.py
def closest_distance(node, nodes):
    dist_2 = np.power(np.sum((nodes - node)**2, axis=1),1/2)
    return np.min(dist_2)

def hausdorff_distance_95(y_true, y_pred):
    y_true_idx = np.where(y_true>0)
    y_pred_idx = np.where(y_pred>0)
    y_true_idx = np.array([y_true_idx[0], y_true_idx[1]])
    y_true_idx = y_true_idx.transpose((1,0))
    y_pred_idx = np.array([y_pred_idx[0], y_pred_idx[1]])
    y_pred_idx = y_pred_idx.transpose((1,0))
    common = y_true*y_pred
    total_common = np.sum(common==1)
    y_true0 = y_true-common
    y_pred0 = y_pred-common
    y_true0_idx = np.where(y_true0>0)
    y_pred0_idx = np.where(y_pred0>0)
    y_true0_idx = np.array([y_true0_idx[0], y_true0_idx[1]])
    y_true0_idx = y_true0_idx.transpose((1,0))
    y_pred0_idx = np.array([y_pred0_idx[0], y_pred0_idx[1]])
    y_pred0_idx = y_pred0_idx.transpose((1,0))
    all_dist_y_true_q = np.hstack([np.array([closest_distance(node, y_pred_idx) for node in y_true0_idx]), np.zeros(total_common)])
    all_dist_y_pred_p = np.hstack([np.array([closest_distance(node, y_true_idx) for node in y_pred0_idx]), np.zeros(total_common)])
    undirected_h_d_95 = (np.percentile(all_dist_y_true_q,95) + np.percentile(all_dist_y_pred_p,95))/2
    return undirected_h_d_95



