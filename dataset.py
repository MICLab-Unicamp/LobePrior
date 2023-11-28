#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import torch
import numpy as np
import torchio as tio
import SimpleITK as sitk
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

from utils.to_onehot import to_onehot
from utils.transform3D import random_crop

RAW_DATA_FOLDER = os.getenv("HOME")
RAW_DATA_FOLDER_ISOMETRIC = os.path.join(RAW_DATA_FOLDER, "DataSets/outputs_registered_high/isometric_cliped_and_normalized")

TEMPLATE_1_PATH  = os.path.join(RAW_DATA_FOLDER_ISOMETRIC, 'model_fusion/group_1.npz')
TEMPLATE_2_PATH  = os.path.join(RAW_DATA_FOLDER_ISOMETRIC, 'model_fusion/group_2.npz')
TEMPLATE_3_PATH  = os.path.join(RAW_DATA_FOLDER_ISOMETRIC, 'model_fusion/group_3.npz')
TEMPLATE_4_PATH  = os.path.join(RAW_DATA_FOLDER_ISOMETRIC, 'model_fusion/group_4.npz')
TEMPLATE_5_PATH  = os.path.join(RAW_DATA_FOLDER_ISOMETRIC, 'model_fusion/group_5.npz')
TEMPLATE_6_PATH  = os.path.join(RAW_DATA_FOLDER_ISOMETRIC, 'model_fusion/group_6.npz')
TEMPLATE_7_PATH  = os.path.join(RAW_DATA_FOLDER_ISOMETRIC, 'model_fusion/group_7.npz')
TEMPLATE_8_PATH  = os.path.join(RAW_DATA_FOLDER_ISOMETRIC, 'model_fusion/group_8.npz')
TEMPLATE_9_PATH  = os.path.join(RAW_DATA_FOLDER_ISOMETRIC, 'model_fusion/group_9.npz')
TEMPLATE_10_PATH = os.path.join(RAW_DATA_FOLDER_ISOMETRIC, 'model_fusion/group_10.npz')
TEMPLATE_11_PATH = os.path.join(RAW_DATA_FOLDER_ISOMETRIC, 'model_fusion/group_11.npz')

def showImages(image, label):
	f, (plot1, plot2, plot3, plot4, plot5, plot6) = plt.subplots(1, 6, figsize = (12, 6))
	plot1.imshow(image[0][image.shape[1]//2])
	plot1.set_axis_off()
	plot2.imshow(label[0][label.shape[1]//2])
	plot2.set_axis_off()
	plot3.imshow(image[0][:,image.shape[1]//2])
	plot3.set_axis_off()
	plot4.imshow(label[0][:,label.shape[1]//2])
	plot4.set_axis_off()
	plot5.imshow(image[0][:,:,image.shape[1]//2])
	plot5.set_axis_off()
	plot6.imshow(label[0][:,:,label.shape[1]//2])
	plot6.set_axis_off()
	plt.show()
	plt.close()

class CTDataset3D(Dataset):
	def __init__(self, mode, labels_name=[1,2,3,4,5], bronchi=False, transforms=None):
		self.bronchi = bronchi
		self.mode = mode
		self.labels_name = labels_name
		self.dataset = sorted(glob.glob(os.path.join(RAW_DATA_FOLDER_ISOMETRIC, mode, "*.npz")))

		print('\tFolder:', RAW_DATA_FOLDER_ISOMETRIC)
		print('\tTamanho do dataset ({}): {}'.format(mode, len(self.dataset)))
		print('\tMode:', self.mode)

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, i):
		npz_path = self.dataset[i]
		npz = np.load(npz_path)
		img, tgt = npz["image"][:].astype(np.float32), npz["label"][:].astype(np.float32)

		ID = os.path.basename(npz_path).replace('.npz','').replace('_affine3D','').replace('_rigid3D','')

		##########################################################################
		subject = tio.Subject(
			image=tio.ScalarImage(tensor = img),
			label=tio.LabelMap(tensor = tgt),
		)
		transform = tio.Resize((64, 64, 64))
		transformed = transform(subject)
		img_high = transformed.image.numpy()
		tgt_high = transformed.label.numpy()
		##########################################################################

		img_high = torch.from_numpy(img_high).float()
		tgt_high = torch.from_numpy(tgt_high).float()

		segmentation = np.array(tgt.squeeze().argmax(axis=0)).astype(np.uint8)
		if segmentation.max()>8:
			segmentation[segmentation>8]=0
		segmentation[segmentation>0]=1

		new_image = np.zeros(img[0].shape).astype(img.dtype)
		new_image = np.where(segmentation == 1, img, img.min())
		img = new_image

		if self.mode=='train':
			img, tgt = random_crop(img, tgt, 64, 128, 128)

		img = torch.from_numpy(img).float()
		tgt = torch.from_numpy(tgt).float()

		return {"image_h": img_high, "label_h": tgt_high, "image": img, "label": tgt, "npz_path":npz_path, "ID":ID}

def buscaImagesByGoup(GROUP):
	images_all = glob.glob(os.path.join(RAW_DATA_FOLDER_ISOMETRIC, 'groups','group_'+str(GROUP),'npz_rigid/*.npz'))
	#print(len(images_all))
	return [os.path.basename(npz_path).replace("_affine3D.npz", '').replace(".npz", '') for npz_path in images_all]

class CTDataset3DWithTemplate(Dataset):
	def __init__(self, mode, labels_name=[1,2,3,4,5], bronchi=False, transforms=None):
		self.bronchi = bronchi
		self.mode = mode
		self.labels_name = labels_name
		self.dataset = sorted(glob.glob(os.path.join(RAW_DATA_FOLDER_ISOMETRIC, 'npz_rigid', mode, "*.npz")))

		self.group_1 = buscaImagesByGoup(GROUP = 1)
		self.group_2 = buscaImagesByGoup(GROUP = 2)
		self.group_3 = buscaImagesByGoup(GROUP = 3)
		self.group_4 = buscaImagesByGoup(GROUP = 4)
		self.group_5 = buscaImagesByGoup(GROUP = 5)
		self.group_6 = buscaImagesByGoup(GROUP = 6)
		self.group_7 = buscaImagesByGoup(GROUP = 7)
		self.group_8 = buscaImagesByGoup(GROUP = 8)
		self.group_9 = buscaImagesByGoup(GROUP = 9)
		self.group_10 = buscaImagesByGoup(GROUP = 10)
		self.group_11 = buscaImagesByGoup(GROUP = 11)

		print('\tFolder:', RAW_DATA_FOLDER_ISOMETRIC)
		print('\tTamanho do dataset ({}): {}'.format(mode, len(self.dataset)))
		print('\tMode:', self.mode)

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, i):
		npz_path = self.dataset[i]
		npz = np.load(npz_path)
		img, tgt = npz["image"][:].astype(np.float32), npz["label"][:].astype(np.float32)
		ID = os.path.basename(npz_path).replace('.npz','').replace('_affine3D','').replace('_rigid3D','')

		img = img.transpose(2,1,0)
		tgt = tgt.transpose(2,1,0)

		if len(tgt.shape)==3:
			if tgt.max()==8 or tgt.max()==520:
				mask_one, onehot, labels_one = to_onehot(tgt, [7,8,4,5,6], single_foregound_lable=False, onehot_type=np.dtype(np.int8))
			elif tgt.max()==5:
				mask_one, onehot, labels_one = to_onehot(tgt, [1,2,3,4,5], single_foregound_lable=False, onehot_type=np.dtype(np.int8))

			tgt = onehot
			img = np.expand_dims(img, 0)

		my_ID = []
		my_ID.append(ID)

		if my_ID[0] in self.group_1:
			template = np.load(TEMPLATE_1_PATH)["model"][:].astype(np.float32)
		elif my_ID[0] in self.group_2:
			template = np.load(TEMPLATE_2_PATH)["model"][:].astype(np.float32)
		elif my_ID[0] in self.group_3:
			template = np.load(TEMPLATE_3_PATH)["model"][:].astype(np.float32)
		elif my_ID[0] in self.group_4:
			template = np.load(TEMPLATE_4_PATH)["model"][:].astype(np.float32)
		elif my_ID[0] in self.group_5:
			template = np.load(TEMPLATE_5_PATH)["model"][:].astype(np.float32)
		elif my_ID[0] in self.group_6:
			template = np.load(TEMPLATE_6_PATH)["model"][:].astype(np.float32)
		elif my_ID[0] in self.group_7:
			template = np.load(TEMPLATE_7_PATH)["model"][:].astype(np.float32)
		elif my_ID[0] in self.group_8:
			template = np.load(TEMPLATE_8_PATH)["model"][:].astype(np.float32)
		elif my_ID[0] in self.group_9:
			template = np.load(TEMPLATE_9_PATH)["model"][:].astype(np.float32)
		elif my_ID[0] in self.group_10:
			template = np.load(TEMPLATE_10_PATH)["model"][:].astype(np.float32)
		elif my_ID[0] in self.group_11:
			template = np.load(TEMPLATE_11_PATH)["model"][:].astype(np.float32)
		else:
			print('Template model não expecificado.')

		##########################################################################

		subject = tio.Subject(
			image=tio.ScalarImage(tensor = img),
			label=tio.LabelMap(tensor = tgt),
			template=tio.LabelMap(tensor = template),
		)
		transform = tio.Resize((128, 128, 128))
		transformed = transform(subject)
		img_high = transformed.image.numpy()
		tgt_high = transformed.label.numpy()
		template_high = transformed.template.numpy()

		#assert template_high.shape == tgt_high.shape, 'Template and label should be same shape, instead are {}, {}'.format(template.shape, tgt.shape)
		#assert len(template.shape) == 4, 'Inputs should be B*W*H*N tensors, instead have shape {}'.format(img.shape)

		#showImages(img_high, tgt_high)

		##########################################################################

		if len(tgt.shape)==3:
			if tgt.max()==8 or tgt.max()==520:
				mask_one, onehot, labels_one = to_onehot(tgt, [7,8,4,5,6], single_foregound_lable=False, onehot_type=np.dtype(np.int8))
			elif tgt.max()==5:
				mask_one, onehot, labels_one = to_onehot(tgt, [1,2,3,4,5], single_foregound_lable=False, onehot_type=np.dtype(np.int8))

			tgt = onehot
			img = np.expand_dims(img, 0)

		assert tgt.shape==template.shape, f'Label and template with different shapes: {tgt.shape} and {template.shape}'

		#showImages(img, tgt)

		##########################################################################

		segmentation = np.array(tgt.squeeze().argmax(axis=0)).astype(np.uint8)
		if segmentation.max()>8:
			segmentation[segmentation>8]=0
		segmentation[segmentation>0]=1

		new_image = np.zeros(img[0].shape).astype(img.dtype)
		new_image = np.where(segmentation == 1, img, img.min())
		img = new_image

		#showImages(img, tgt)
		#showImages(img, template)

		##########################################################################

		img_high = torch.from_numpy(img_high).float()
		tgt_high = torch.from_numpy(tgt_high).float()
		template_high = torch.from_numpy(template_high).float()
		img = torch.from_numpy(img).float()
		tgt = torch.from_numpy(tgt).float()
		template = torch.from_numpy(template).float()

		return {"image_h": img_high, "label_h": tgt_high, 'template_high': template_high, "image": img, "label": tgt, "template": template, "npz_path": npz_path, "ID":ID}
