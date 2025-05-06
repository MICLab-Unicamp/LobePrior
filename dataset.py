#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import torch
import random

import numpy as np
import torchio as tio
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from utils.to_onehot import mask_to_onehot
from utils.transform3D import TransformsnnUNet

RAW_DATA_FOLDER = '/mnt/data/registered_images_no_dice'
RAW_DATA_FOLDER_MODEL_FUSION = '/mnt/data/registered_images_no_dice/model_fusion'

NOMES = [
			'No',
			'covid19severity_1063',
			'covid19severity_1107',
			'covid19severity_1128',
			'covid19severity_183',
			'covid19severity_414',
			'covid19severity_693',
			'covid19severity_6',
			'No',
			'covid19severity_749',
			'covid19severity_892',
			'covid19severity_947',
			'covid19severity_97',
			'No',
		]

def insere_lesao_blended(img, lung, img_lesion, lesion):

	sigma=2
	# Aplica Gaussian blur à máscara para criar uma borda suave
	soft_mask = gaussian_filter(lesion.astype(np.float32), sigma=sigma)

	# Normaliza a máscara para ficar entre 0 e 1
	soft_mask = np.clip(soft_mask, 0, 1)

	# Combina as duas imagens usando a máscara suave (blending linear)
	blended_image = (1 - soft_mask) * img + soft_mask * img_lesion

	return blended_image

class CTDataset3DTemplateAirway(Dataset):
	def __init__(self, mode, transforms=None):
		self.mode = mode
		self.transform = TransformsnnUNet(verbose=False)
		self.dataset = sorted(glob.glob(os.path.join(RAW_DATA_FOLDER, mode, '*/*.npz')))
		print(len(self.dataset))

		print('\tTamanho do dataset ({}): {}'.format(mode, len(self.dataset)))
		print('\tMode:', self.mode)

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, i):
		npz_path = self.dataset[i]
		npz = np.load(npz_path)
		img, tgt, airway = npz["image"][:].astype(np.float32), npz["label"][:].astype(np.float32), npz["airway"][:].astype(np.float32)

		ID_image = os.path.basename(npz_path).replace('.npz','').replace('_affine3D','').replace('_rigid3D','')

		group = npz["group"]
		npz_template_path = os.path.join(RAW_DATA_FOLDER_MODEL_FUSION, 'group_'+str(group)+'.npz')
		template = np.load(npz_template_path)["model"][:].astype(np.float32)

		img = img.transpose(2,1,0)
		tgt = tgt.transpose(2,1,0)
		airway = airway.transpose(2,1,0)
		#template = template.transpose(2,1,0)

		lung = tgt.copy()
		lung[lung>0]=1

		ID_lesion = random.choice(NOMES)

		if ID_lesion!='No':
			npz_lesion_path = os.path.join(RAW_DATA_FOLDER_MODEL_FUSION, 'groups/group_'+str(group)+'/npz_rigid', ID_lesion+'_rigid3D.npz')
			#print(npz_lesion_path)

			npz_lesion = np.load(npz_lesion_path)
			img_lesion, lung_lesion, lesion = npz_lesion["image"][:].astype(np.float32), npz_lesion["lung"][:].astype(np.float32), npz_lesion["lesion"][:].astype(np.float32)

			img_lesion = img_lesion.transpose(2,1,0)
			lung_lesion = lung_lesion.transpose(2,1,0)
			lesion = lesion.transpose(2,1,0)

			assert img.shape==lung_lesion.shape==img_lesion.shape==lesion.shape, f'{img_lesion.shape}, {lung_lesion.shape}, {lesion.shape}'

			lesion = lesion*lung

			img = insere_lesao_blended(img, lung, img_lesion, lesion)

		tgt = mask_to_onehot(tgt)
		#tgt_high = mask_to_onehot(tgt_high)
		#airway = mask_to_onehot(airway)

		if len(img.shape)==3:
			img = np.expand_dims(img, 0)
		if len(airway.shape)==3:
			airway = np.expand_dims(airway, 0)

		#print(img.shape, tgt.shape, airway.shape, template.shape)

		if (img.shape[1]*img.shape[2]*img.shape[3])>60000000:
			subject = tio.Subject(
					image=tio.ScalarImage(tensor = img),
					label=tio.LabelMap(tensor = tgt),
					#lung=tio.LabelMap(tensor = lung),
					airway=tio.LabelMap(tensor = airway),
					template=tio.LabelMap(tensor = template),
			)
			transform = tio.Resize((350, 400, 400))
			#transform = tio.Resize((400, 400, 350))
			transformed = transform(subject)

			img = transformed.image.numpy()
			tgt = transformed.label.numpy()
			#lung = transformed.lung.numpy()
			airway = transformed.airway.numpy()
			template = transformed.template.numpy()

		subject = tio.Subject(
			image=tio.ScalarImage(tensor = img),
			label=tio.LabelMap(tensor = tgt),
			#lung=tio.LabelMap(tensor = lung),
			airway=tio.LabelMap(tensor = airway),
			#template=tio.LabelMap(tensor = template),
		)
		transform = tio.Resize((128, 128, 128))
		transformed = transform(subject)
		img_high = transformed.image.numpy()
		tgt_high = transformed.label.numpy()
		#lung_high = transformed.lung.numpy()
		airway_high = transformed.airway.numpy()
		#template_high = transformed.template.numpy()

		#segmentation = np.array(tgt.squeeze().argmax(axis=0)).astype(np.uint8)
		#if segmentation.max()>8:
		#	segmentation[segmentation>8]=0
		#segmentation[segmentation>0]=1

		#new_image = np.zeros(img[0].shape).astype(img.dtype)
		#new_image = np.where(segmentation == 1, img, img.min())
		#img = new_image

		img_high = torch.from_numpy(img_high).float()
		tgt_high = torch.from_numpy(tgt_high).float()
		#lung_high = torch.from_numpy(lung_high).float()
		img = torch.from_numpy(img).float()
		tgt = torch.from_numpy(tgt).float()
		#lung = torch.from_numpy(lung).float()
		airway = torch.from_numpy(airway).float()
		template = torch.from_numpy(template).float()

		if self.mode=='train':
			if self.transform is not None:
				img, tgt = self.transform(img, tgt)

		assert img[0,0].shape==tgt[0,0].shape==airway[0,0].shape==template[0,0].shape, f'Imagem e label {ID_image} no grupo {group} com tamanho de shapes diferentes: {img.shape} {tgt.shape} {airway.shape} {template.shape}'
		assert len(img.shape)==len(tgt.shape)==len(airway.shape)==len(template.shape), f'Imagem e label {ID_image} no grupo {group} com tamanho de shapes diferentes: {len(img.shape)} {len(tgt.shape)} {len(airway.shape)} {len(template.shape)}'

		return {"image_h": img_high, "label_h": tgt_high, "airway_h": airway_high, "image": img, "label": tgt, 'airway': airway, "template": template, 'group': group}
