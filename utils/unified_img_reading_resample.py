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

DATA_FOLDER_NPZ = '../results/output_convert/npz'
DATA_FOLDER_IMAGE = '../results/output_convert/images'
DATA_FOLDER_LABEL = '../results/output_convert/labels'
RAW_DATA_FOLDER = "../../DataSets/raw"

def show_image_label_to_one_hot(image, label, n_slice=None):
	if n_slice is None:
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

class Clip():
	def __init__(self, min, max):
		self.min = min
		self.max = max

	def __call__(self, x_y: Tuple[np.ndarray, np.ndarray]):
		x, y = x_y
		x = np.clip(x, self.min, self.max)
		return x, y

class MinMaxNormalize():
	def __init__(self, vmin=-1024, vmax=600):
		self.vmin = vmin
		self.vmax = vmax

	def __call__(self, x_y: Tuple[np.ndarray, np.ndarray]):
		x, y = x_y
		x = (x - x.min()) / (x.max() - x.min())
		#x = (x - self.vmin)/(self.vmax - self.vmin)
		return x, y

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

def resample_img(itk_image, out_spacing=[2.0, 2.0, 2.0], is_label=False):

    # Resample images to 2mm spacing with SimpleITK
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)

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

def unified_img_reading(image, mask=None, spacing=[1.0, 1.0, 1.0], torch_convert=False, isometric=False, convert_to_onehot=None):
	'''
	path: path to main image
	mask_path: path to segmentation image if available
	torch_convert: converts to torch format and expands channel dimension
	isometric: puts image to 1, 1, 1 voxel size

	returns: data, mask, spacing
	'''
	spacing = image.GetSpacing()

	# Reorder by slice location and remove non aligned shapes
	#if isinstance(path, list):
	#	filter_dicom_list(path)

	spacing = image.GetSpacing()[::-1]
	data = sitk.GetArrayFromImage(image)
	directions = np.asarray(image.GetDirection())
	#mask = None
	if len(directions) == 9:
		data = np.flip(data, np.where(directions[[0,4,8]][::-1]<0)[0]).copy()  # cryptic one liner from lungmask
		if mask is not None:
			mask = sitk.GetArrayFromImage(mask)
			mask = np.flip(mask, np.where(directions[[0,4,8]][::-1]<0)[0]).copy()
			mask = corrige_label(mask)

	if torch_convert:
		if data.dtype == "uint16":
			data = data.astype(np.int16)
		data = torch.from_numpy(data).float()
		if len(data.shape) == 3:
			data = data.unsqueeze(0)
		if mask is not None:
			if mask.dtype == "uint16":
				mask = mask.astype(np.int16)
			if convert_to_onehot is not None:
				if len(mask.shape) > 3:
					raise ValueError(f"Are you sure you want to convert to one hot a mask that is of this shape: {mask.shape}")
				mask = int_to_onehot(mask, overhide_max=convert_to_onehot)
			mask = torch.from_numpy(mask).float()

	return data, mask, spacing

def unified_img_reading_isometric(image, mask=None, spacing=[1.0, 1.0, 1.0], torch_convert=False, isometric=False, convert_to_onehot=None, show_message=True):
	'''
	path: path to main image
	mask_path: path to segmentation image if available
	torch_convert: converts to torch format and expands channel dimension
	isometric: puts image to 1, 1, 1 voxel size

	returns: data, mask, spacing
	'''
	spacing = image.GetSpacing()

	# Reorder by slice location and remove non aligned shapes
	#if isinstance(path, list):
	#	filter_dicom_list(path)

	spacing = image.GetSpacing()[::-1]
	data = sitk.GetArrayFromImage(image)
	directions = np.asarray(image.GetDirection())
	#mask = None
	if len(directions) == 9:
		data = np.flip(data, np.where(directions[[0,4,8]][::-1]<0)[0]).copy()  # cryptic one liner from lungmask
		if mask is not None:
			mask = sitk.GetArrayFromImage(mask)
			mask = np.flip(mask, np.where(directions[[0,4,8]][::-1]<0)[0]).copy()
			mask = corrige_label(mask)

	if isometric:
		if show_message:
			print(f"Pre isometry stats {spacing}: {data.shape} max {data.max()} min {data.min()}")
		data = zoom(data, spacing)
		if mask is not None:
			mask = zoom(mask, spacing, order=0)
			mask_shape = mask.shape
		else:
			mask_shape = None
		if show_message:
			print(f"Post isometry stats {spacing} ->~ [1, 1, 1]: {data.shape}/{mask_shape} max {data.max()} min {data.min()}")
		spacing = [1.0, 1.0, 1.0]

	if torch_convert:
		if data.dtype == "uint16":
			data = data.astype(np.int16)
		data = torch.from_numpy(data).float()
		if len(data.shape) == 3:
			data = data.unsqueeze(0)
		if mask is not None:
			if mask.dtype == "uint16":
				mask = mask.astype(np.int16)
			if convert_to_onehot is not None:
				if len(mask.shape) > 3:
					raise ValueError(f"Are you sure you want to convert to one hot a mask that is of this shape: {mask.shape}")
				mask = int_to_onehot(mask, overhide_max=convert_to_onehot)
			mask = torch.from_numpy(mask).float()

	return data, mask, spacing

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

def main(args):
	transform = torchvision.transforms.Compose([CTHUClip(-1024, 600)])

	torch_convert = False
	datasets = ['luna16','coronacases']

	for dataset in datasets:
		for mode in ['train','val','test']:
			if torch_convert:
				os.makedirs(os.path.join(DATA_FOLDER_NPZ, mode), exist_ok=True)
			else:
				os.makedirs(os.path.join(DATA_FOLDER_IMAGE, mode), exist_ok=True)
				os.makedirs(os.path.join(DATA_FOLDER_LABEL), exist_ok=True)

			if dataset=='coronacases':
				all_images = sorted(glob.glob(os.path.join(RAW_DATA_FOLDER, 'covid19-ct-scans/COVID-19-CT-Seg_8cases', mode, '*.nii.gz')))
			elif dataset=='luna16':
				all_images = sorted(glob.glob(os.path.join(RAW_DATA_FOLDER, 'images_35x5x10', mode, "*.mhd")))
			print(f'\nTamanho do dataset ({mode}): {len(all_images)}\n')

			for image_path in all_images:
				ID_image = os.path.basename(image_path).replace(".nii.gz", '').replace(".mhd", '')
				if dataset=='coronacases':
					label_path = os.path.join(RAW_DATA_FOLDER, 'covid19-ct-scans/COVID lung lobe segmentation_labels', ID_image+'.nii.gz')
				elif dataset=='luna16':
					label_path = os.path.join(RAW_DATA_FOLDER, 'annotations', ID_image+"_LobeSegmentation.nrrd")

				image = sitk.ReadImage(image_path)
				label = sitk.ReadImage(label_path)
				print(sitk.GetArrayFromImage(image).shape, sitk.GetArrayFromImage(label).shape)
				image = resample_img(image, out_spacing=[1.4, 1.4, 2.5], is_label=False)
				label = resample_img(label, out_spacing=[1.4, 1.4, 2.5], is_label=True)
				#image.CopyInformation(sitk.ReadImage(sitk.ReadImage(image_path)))
				#label.CopyInformation(sitk.ReadImage(sitk.ReadImage(label_path)))
				image.SetSpacing(np.array([1.4, 1.4, 2.5]).astype(float))
				label.SetSpacing(np.array([1.4, 1.4, 2.5]).astype(float))
				print(sitk.GetArrayFromImage(image).shape, sitk.GetArrayFromImage(label).shape)

				if torch_convert:
					image, label, spacing = unified_img_reading(image, mask=label, isometric=True, torch_convert=True, convert_to_onehot=6)
					print(image.shape, label.shape, spacing)
					#show_image_label_to_one_hot(image, label)
					#show_image_to_one_hot_axial_coronal_sagital(image[0])

					image, label = transform((image, label))
					#print(image.min(), image.max())
					#print(label.min(), label.max())

					save_path = os.path.join(DATA_FOLDER_NPZ, mode, f'{ID_image}.npz')
					np.savez_compressed(save_path, image=image, label=label, ID=ID_image, ct_path=image_path, mask_path=label_path, target="lobes")

					print('\n')
				else:

					image, label, spacing = unified_img_reading(image, mask=label, isometric=True, torch_convert=False, convert_to_onehot=6)

					image, label = transform((image, label))
					print(image.min(), image.max())
					print(label.min(), label.max())

					output_image = sitk.GetImageFromArray(image)
					output_image.SetSpacing(np.array([1.4, 1.4, 2.5]).astype(float))
					sitk.WriteImage(output_image, os.path.join(DATA_FOLDER_IMAGE, mode, ID_image+'.nii.gz'))
					output_label = sitk.GetImageFromArray(label)
					output_label.SetSpacing(np.array([1.4, 1.4, 2.5]).astype(float))
					sitk.WriteImage(output_label, os.path.join(DATA_FOLDER_LABEL, ID_image+'.nii.gz'))

if __name__ == '__main__':
	os.system('cls' if os.name == 'nt' else 'clear')
	sys.exit(main(sys.argv))
