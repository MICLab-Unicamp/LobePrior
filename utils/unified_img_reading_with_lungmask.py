#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import torch
import pydicom
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

from scipy.ndimage import zoom
from collections import Counter
from typing import List
from typing import Tuple
from lungmask import mask as lungmask

DATA_FOLDER = os.getenv("HOME")
DATA_FOLDER_NPZ = '../results/converted_images/npz'
DATA_FOLDER_IMAGE = '../results/converted_images/images'
DATA_FOLDER_LABEL = '../results/converted_images/labels'
DATA_FOLDER_LUNG = '../results/converted_images/lungs'

def show_image_label_to_one_hot(image, label):
	n_slice = image.shape[1]//2

	f, (plot0, plot1, plot2, plot3, plot4, plot5, plot6) = plt.subplots(1, 7, figsize = (12, 6))
	plot0.imshow(image[0,n_slice]), plot0.set_axis_off()
	plot1.imshow(label[0,n_slice]), plot1.set_axis_off()
	plot2.imshow(label[1,n_slice]), plot2.set_axis_off()
	plot3.imshow(label[2,n_slice]), plot3.set_axis_off()
	plot4.imshow(label[3,n_slice]), plot4.set_axis_off()
	plot5.imshow(label[4,n_slice]), plot5.set_axis_off()
	plot6.imshow(label[5,n_slice]), plot6.set_axis_off()
	plt.show()
	plt.close()

def show_image_to_one_hot_axial_coronal_sagital(image):
	print(image.shape)

	f, (plot0, plot1, plot2) = plt.subplots(1, 3, figsize = (12,6))
	plot0.imshow(image[image.shape[0]//2]), plot0.set_axis_off()
	plot1.imshow(image[:,image.shape[1]//2]), plot1.set_axis_off()
	plot2.imshow(image[:,:,image.shape[2]//2]), plot2.set_axis_off()
	plt.show()
	plt.close()

def corrige_label(label):
	if label.max()==8 or label.max()==520:
		label[label == 7] = 1
		label[label == 8] = 2
		label[label == 4] = 3
		label[label == 5] = 4
		label[label == 6] = 5

		label[label > 6] = 0

	return label

def all_idx(idx, axis):
	grid = np.ogrid[tuple(map(slice, idx.shape))]
	grid.insert(axis, idx)
	return tuple(grid)

def int_to_onehot(matrix, overhide_max=None):
	'''
	Converts a matrix of int values (will try to convert) to one hot vectors
	'''
	if overhide_max is None:
		vec_len = int(matrix.max() + 1)
	else:
		vec_len = overhide_max

	onehot = np.zeros((vec_len,) + matrix.shape, dtype=int)

	int_matrix = matrix.astype(int)
	onehot[all_idx(int_matrix, axis=0)] = 1

	return onehot

def filter_dicom_list(path: List):
	initial_l = len(path)
	ps_dcms = [(p, pydicom.read_file(p)) for p in path]

	# Remove no slice location
	pre_len = len(ps_dcms)
	ps_dcms = [pydicom_tuple for pydicom_tuple in ps_dcms if hasattr(pydicom_tuple[1], "SliceLocation")]
	diff = pre_len - len(ps_dcms)
	if diff > 0:
		print(f"WARNING: Removed {diff} slices from series due to not having SliceLocation")

	# Order by slice location
	ps_dcms = sorted(ps_dcms, key=lambda s: s[1].SliceLocation)

	# Select most common shape
	ps_dcms_shapes = [(p, dcm, (dcm.Rows, dcm.Columns)) for p, dcm in ps_dcms]
	most_common_shape = Counter([shape for _, _, shape in ps_dcms_shapes]).most_common(1)[0][0]
	path = [(p, dcm) for p, dcm, dcm_shape in ps_dcms_shapes if dcm_shape == most_common_shape]
	path_diff = initial_l - len(path)
	if path_diff != 0:
		print(f"WARNING: {path_diff} slices removed due to misaligned shape.")
	ps_dcms = [(p, dcm) for p, dcm, _ in ps_dcms_shapes]

	# Select consistent SpacingBetweenSlices or SliceThickness
	initial_l = len(path)
	try:
		ps_dcms_spacings = [(p, dcm, dcm.SpacingBetweenSlices) for p, dcm in ps_dcms]
		attr = "spacing"
	except AttributeError:
		ps_dcms_spacings = [(p, dcm, dcm.SliceThickness) for p, dcm in ps_dcms]
		attr = "thickness"
	most_common_spacing = Counter([spacing for _, _, spacing in ps_dcms_spacings]).most_common(1)[0][0]
	path = [p for p, _, spacing in ps_dcms_spacings if spacing == most_common_spacing]
	path_diff = initial_l - len(path)
	if path_diff != 0:
		print(f"WARNING: {path_diff} slices removed due to inconsistent {attr}.")

	return path

class CTHUClip():
	'''
	Clip and normalize to [0-1] range, taking into account a constant intended maximum and minimum value
	regardless of what is present in the image
	'''
	def __init__(self, vmin=-1024, vmax=600):
		self.vmin = vmin
		self.vmax = vmax

	def __call__(self, x_y: Tuple[np.ndarray, np.ndarray]):
		x, y = x_y
		if torch.is_tensor(x):
			x = torch.clip(x, self.vmin, self.vmax)
		elif isinstance(x, np.ndarray):
			x = np.clip(x, self.vmin, self.vmax)
		else:
			raise ValueError(f"Unsupported x type for CTHUClip {type(x)}")

		x = (x - self.vmin)/(self.vmax - self.vmin)

		if y is not None:
			return x, y
		else:
			return x

	def __str__(self):
		return f"CTHUClip vmin: {self.vmin} vmax: {self.vmax}"

def unified_img_reading_lungmask(path, mask_path=None, torch_convert=False, isometric=False, convert_to_onehot=None, show_message=True):
	'''
	path: path to main image
	mask_path: path to segmentation image if available
	torch_convert: converts to torch format and expands channel dimension
	isometric: puts image to 1, 1, 1 voxel size

	returns: data, mask, spacing
	'''
	# Reorder by slice location and remove non aligned shapes
	if isinstance(path, list):
		filter_dicom_list(path)

	image = sitk.ReadImage(path)
	spacing = image.GetSpacing()[::-1]
	data = sitk.GetArrayFromImage(image)
	directions = np.asarray(image.GetDirection())
	mask = None
	if len(directions) == 9:
		data = np.flip(data, np.where(directions[[0,4,8]][::-1]<0)[0]).copy()  # cryptic one liner from lungmask
		if mask_path is not None:
			mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
			mask = np.flip(mask, np.where(directions[[0,4,8]][::-1]<0)[0]).copy()
			mask = corrige_label(mask)
			mask_lung = lungmask.apply( data )
		else:
			mask_lung = lungmask.apply( data )

	if isometric:
		if show_message:
			print(f"Pre isometry stats {spacing}: {data.shape} max {data.max()} min {data.min()}")
		data = zoom(data, spacing)
		if mask is not None:
			mask = zoom(mask, spacing, order=0)
			mask_lung = zoom(mask_lung, spacing, order=0)
			mask_shape = mask.shape
			mask_lung_shape = mask.shape
		else:
			mask_shape = None
			mask_lung_shape = None
		if show_message:
			print(f"Post isometry stats {spacing} ->~ [1, 1, 1]: {data.shape}/{mask_shape} {data.shape}/{mask_lung_shape} max {data.max()} min {data.min()}")
		spacing = [1.0, 1.0, 1.0]

	if torch_convert:
		if data.dtype == "uint16":
			data = data.astype(np.int16)
		data = torch.from_numpy(data).float()
		if len(data.shape) == 3:
			data = data.unsqueeze(0)
		if mask_path is not None:
			if mask.dtype == "uint16":
				mask = mask.astype(np.int16)
			if mask_lung.dtype == "uint16":
				mask_lung = mask.astype(np.int16)
			if convert_to_onehot is not None:
				if len(mask.shape) > 3:
					raise ValueError(f"Are you sure you want to convert to one hot a mask that is of this shape: {mask.shape}")
				mask = int_to_onehot(mask, overhide_max=convert_to_onehot)
			mask = torch.from_numpy(mask).float()
			mask_lung = torch.from_numpy(mask_lung).float()

	return data, mask, mask_lung, spacing

def main(args):
	torch_convert = False

	transform = torchvision.transforms.Compose([CTHUClip(-1024, 600)])

	for mode in ['coronacases']:
		all_images = sorted(glob.glob(os.path.join(DATA_FOLDER, 'dados/dados', mode, '*.nii.gz')))
		print(f'\nTamanho do dataset ({mode}): {len(all_images)}\n')

		if torch_convert:
			os.makedirs(os.path.join(DATA_FOLDER_NPZ, mode), exist_ok=True)
		else:
			os.makedirs(os.path.join(DATA_FOLDER_IMAGE, mode), exist_ok=True)
			os.makedirs(os.path.join(DATA_FOLDER_LABEL, mode), exist_ok=True)
			os.makedirs(os.path.join(DATA_FOLDER_LUNG, mode), exist_ok=True)

		for image_path in all_images:
			ID_image = os.path.basename(image_path).replace(".nii.gz", '').replace(".mhd", '')
			label_path = os.path.join(DATA_FOLDER, 'dados/segmentations/lungmask/LTRCLobes_R231', ID_image+'.nii.gz')
			#print(label_path)

			if torch_convert:
				image, label, spacing = unified_img_reading_lungmask(image_path, mask_path=label_path, isometric=True, torch_convert=True, convert_to_onehot=6)
				print(image.shape, label.shape, spacing)
				#show_image_label_to_one_hot(image, label)
				#show_image_to_one_hot_axial_coronal_sagital(image[0])

				if transform:
					image, label = transform((image, label))
				print(image.min(), image.max())
				print(label.min(), label.max())

				save_path = os.path.join(DATA_FOLDER_NPZ, mode, f'{ID_image}.npz')
				np.savez_compressed(save_path, image=image, label=label, lung=lung, ID=ID_image, ct_path=image_path, target="lung")

			else:

				image, label, lung, spacing = unified_img_reading_lungmask(image_path, mask_path=label_path, isometric=True, torch_convert=False, convert_to_onehot=6)
				print(image.shape, label.shape)

				if transform:
					image, label = transform((image, label))
				print(image.min(), image.max())
				print(label.min(), label.max())

				output_image = sitk.GetImageFromArray(image)
				sitk.WriteImage(output_image, os.path.join(DATA_FOLDER_IMAGE, mode, ID_image+'.nii.gz'))
				output_label = sitk.GetImageFromArray(label)
				sitk.WriteImage(output_label, os.path.join(DATA_FOLDER_LABEL, mode, ID_image+'.nii.gz'))
				output_lung = sitk.GetImageFromArray(lung)
				sitk.WriteImage(output_lung, os.path.join(DATA_FOLDER_LUNG, mode, ID_image+'.nii.gz'))

if __name__ == '__main__':
	os.system('cls' if os.name == 'nt' else 'clear')
	sys.exit(main(sys.argv))
