#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
import SimpleITK as sitk
from math import nan
from typing import List, Dict
from collections import defaultdict
from torch.autograd import Variable

def initialize_metrics_dict():
    '''
    Initializes an empty metrics dict to be given to seg_metrics
    '''
    return defaultdict(lambda: defaultdict(list))


def seg_metrics(gts: np.ndarray, preds: np.ndarray, metrics: Dict[str, Dict[str, List[float]]], struct_names=["bg", "healthy", "unhealthy"]):
    '''
    finds overlap and distance measures of two given segmentations.
    "Overlap measures: Dice, FNError, FPError, jaccard, Volume Similarity (SimpleITK) and Volume Similarity(Taha et al)
    "Distance measures: Hausdorff distance and average hausdorff distance
    '''
    assert (len(gts.shape) == len(preds.shape) and 
            isinstance(gts, np.ndarray) and isinstance(preds, np.ndarray) and gts.dtype == np.uint8 and preds.dtype == np.uint8 and
            (gts >= 0).all() and (gts <= 1).all() and (preds <= 1).all() and (preds >= 0).all()), "Malformed input for seg_metrics"
    
    for gt, pred, str_label in zip(gts, preds, struct_names):
        hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
        overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
        
        img_gt_sitk = sitk.GetImageFromArray(gt)
        img_pred_sitk = sitk.GetImageFromArray(pred)
        
        overlap_measures_filter.Execute(img_gt_sitk, img_pred_sitk)
        
        metrics[str_label]["dice"].append(overlap_measures_filter.GetDiceCoefficient())
        metrics[str_label]["false_negative_error"].append(overlap_measures_filter.GetFalseNegativeError())
        metrics[str_label]["false_positive_error"].append(overlap_measures_filter.GetFalsePositiveError())
        metrics[str_label]["jaccard"].append(overlap_measures_filter.GetJaccardCoefficient())
        metrics[str_label]["volume_similarity"].append(overlap_measures_filter.GetVolumeSimilarity())
        metrics[str_label]["abs_volume_similarity"].append(1-abs(overlap_measures_filter.GetVolumeSimilarity())/2)
        
        try:
            hausdorff_distance_filter.Execute(img_gt_sitk, img_pred_sitk)    
            metrics[str_label]["avg_hd"].append(hausdorff_distance_filter.GetAverageHausdorffDistance())
            metrics[str_label]["hd"].append(hausdorff_distance_filter.GetHausdorffDistance())
        except:
            metrics[str_label]["avg_hd"].append(nan)
            metrics[str_label]["hd"].append(nan)

def main(args):
    mask_path = "../../../ct_pretraining-main/demo_data/0cdc88ef01427b83c9a74754f76b91c4acc5ad7f57ff30cd7880e2b4007488c5_mask.nii"
    mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
    mask = mask[np.newaxis, :]
    mask = torch.from_numpy(mask).float()
    mask = Variable(torch.randn(6, 128, 128)).cpu()
    print(mask.shape)

    metrics = initialize_metrics_dict()

    seg_metrics(gts=mask.numpy().astype(np.uint8), preds=mask.numpy().astype(np.uint8), metrics=metrics, struct_names=["background", "healthy", "unhealthy"])

    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
