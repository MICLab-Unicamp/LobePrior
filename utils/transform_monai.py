#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from monai.transforms import (
						AsDiscreted,
						EnsureChannelFirstd,
						Compose,
						CropForegroundd,
						LoadImaged,
						Orientationd,
						RandCropByPosNegLabeld,
						SaveImaged,
						RandAffined,
						AdjustContrastd,
						ScaleIntensityRanged,
						Spacingd,
						SpatialCropd,
						RandFlipd,
						RandSpatialCropSamplesd,
						RandRotated,
						RandGaussianNoised,
						Rand3DElasticd,
						Invertd,
						ToTensord,
					)
from monai.config import KeysCollection
from monai.transforms import (
    MapTransform,
)

class ToOnehot(MapTransform):
    def __init__(self, keys: KeysCollection, labels=[7,8,4,5,6], single_foregound_label=True, background_channel=True, onehot_type=np.dtype(np.float32)):
        self.keys = keys
        self.labels = labels
        self.single_foregound_label = single_foregound_label
        self.background_channel = background_channel
        self.onehot_type = onehot_type

    def __call__(self, data):
        for k in self.keys:
            matrix = np.around(data[k])
            if len(self.labels) == 0:
                self.labels = np.unique(matrix) 
                self.labels = self.labels[1::]
            
            mask = np.zeros(matrix.shape, dtype=self.onehot_type)
            for i, label in enumerate(self.labels):
                mask += ((matrix == label) * (i+1))
          
            if self.single_foregound_label:
                mask = (mask > 0)
                self.labels = [1]
                
            labels_len = len(self.labels)
                
            onehot = np.zeros((labels_len+1,) + matrix.shape, dtype=self.onehot_type)
            for i in range(mask.max()+1):
                onehot[i] = (mask == i)  
                
            if self.background_channel == False:
                onehot = onehot[1::]

            data[k] = onehot
        return data

def get_transform_monai():
	train_transforms = Compose(
		[
			LoadImaged(keys=["image", "label"]),
			EnsureChannelFirstd(keys=["image"]),
			ToOnehot(keys=["label"], labels=[7,8,4,5,6], single_foregound_label=False, onehot_type=np.dtype(np.int8)),
			#ScaleIntensityRanged(keys=["image"], a_min=-1024, a_max=600, b_min=0.0, b_max=1.0, clip=True),
			CropForegroundd(keys=["image", "label"], source_key="image"),
			Orientationd(keys=["image", "label"], axcodes="PLI"),
			Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
			#RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=(96, 96, 96), pos=1, neg=1, num_samples=4, image_key="image", image_threshold=0,),
			#RandRotated(keys=['image', 'label'], range_x=[15.4, 15.4], prob=0.5, mode=('bilinear', 'nearest')),
			RandGaussianNoised(keys='image', prob=0.5),
			#AdjustContrastd(keys='image', gamma=2),
			#Rand3DElasticd(
			#	keys=["image", "label"],
			#	mode=("bilinear", "nearest"),
			#	prob=1.0,
			#	sigma_range=(5, 8),
			#	magnitude_range=(100, 200),
			#	spatial_size=(256, 256, 64),
			#	translate_range=(2, 50, 50),
			#	rotate_range=(np.pi, np.pi / 36, np.pi / 36),
			#	scale_range=(0.15, 0.15, 0.15),
			#	padding_mode="border",
			#),
			# user can also add other random transforms
			RandAffined(
			     keys=['image', 'label'],
			     mode=('bilinear', 'nearest'),
			     prob=0.5,
			     spatial_size=(256, 256, 64),
			     rotate_range=(0, 0, np.pi/25),
			     scale_range=(0.1, 0.1, 0.1)
			),
			RandFlipd(
				keys=["image", "label"],
				spatial_axis=[0],
				prob=0.40,
			),
			RandFlipd(
				keys=["image", "label"],
				spatial_axis=[1],
				prob=0.40,
			),
			RandFlipd(
				keys=["image", "label"],
				spatial_axis=[2],
				prob=0.40,
			),
			#TransposeD(keys=["image", "label"], indices=(0,3,2,1)),
			#ToTensord(keys=["image", "label"]),
		]
	)
	eval_transforms = Compose(
		[
			LoadImaged(keys=["image", "label"]),
			EnsureChannelFirstd(keys=["image"]),
			ToOnehot(keys=["label"], labels=[7,8,4,5,6], single_foregound_label=False, onehot_type=np.dtype(np.int8)),
			#ScaleIntensityRanged(keys=["image"], a_min=-1024, a_max=600, b_min=0.0, b_max=1.0, clip=True),
			CropForegroundd(keys=["image", "label"], source_key="image"),
			Orientationd(keys=["image", "label"], axcodes="PLI"),
			Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
			#TransposeD(keys=["image", "label"], indices=(0,3,2,1)),
			#ToTensord(keys=["image", "label"]),
		]
	)
	return train_transforms, eval_transforms
