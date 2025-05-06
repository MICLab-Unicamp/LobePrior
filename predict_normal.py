#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import torch
import torchvision
import numpy as np
import torchio as tio
import SimpleITK as sitk
import nibabel as nib
import pytorch_lightning as pl
from matplotlib import pyplot as plt
from monai.inferers import sliding_window_inference

from model.unet_diedre import UNet_SeisDecoders
from utils.general import pos_processamento, post_processing_dist_lung, post_processing_lung
from utils.general import unified_img_reading, busca_path, salvaImageRebuilt
from utils.to_onehot import mask_to_onehot
from utils.transform3D import CTHUClip
from predict_lung  import LungModule

HOME = os.getenv("HOME")
TEMP_IMAGES = 'temp_images'

def get_sample(npz_path):
		ID_image = os.path.basename(npz_path).replace('.npz','').replace('_affine3D','').replace('_rigid3D','')
		#print(f'\tImage name: {ID_image}')

		npz = np.load(npz_path)
		img = npz["image"][:].astype(np.float32)
		#print('Shape:', img.shape)
		#print('MinMax:', img.min(), img.max())

		img = img.transpose(2,1,0)

		if len(img.shape)==3:
			img = np.expand_dims(img, 0)

		subject = tio.Subject(
			image=tio.ScalarImage(tensor = img),
		)
		transform = tio.Resize((128, 128, 128))
		transformed = transform(subject)
		img_high = transformed.image.numpy()

		img_high = torch.tensor(img_high, dtype=torch.float32).unsqueeze(dim=0).cuda()
		img = torch.tensor(img, dtype=torch.float32).unsqueeze(dim=0).cuda()

		return {"image_h": img_high, "image": img, "ID_image":ID_image}

class LoberModuleNormal(pl.LightningModule):
	def __init__(self, hparams):
		super().__init__()

		# O nome precisar ser hparams para o PL.
		self.save_hyperparameters(hparams)

		if self.hparams.mode == "segmentation":
			# Hat: opera em low resolution (fov inteiro)
			self.model_low = UNet_SeisDecoders(n_channels=1, n_classes=1, norm="instance", dim='3d', init_channel=16, joany_conv=False, dict_return=False)
			# Seg: opera em high resolution (patch)
			self.model = UNet_SeisDecoders(n_channels=7, n_classes=1, norm="instance", dim='3d', init_channel=16, joany_conv=False, dict_return=False)

	def forward_per_lobe(self, x, y_seg_resize):

		#template = (template > 0.3).float()

		x_new = torch.cat((x, y_seg_resize), dim = 1)

		output_one, output_two, output_three, output_four, output_five, output_lung = sliding_window_inference(
			x_new.cuda(),
			roi_size=(128, 128, 128),
			sw_batch_size=1,
			predictor=self.model.cuda(),
			#overlap=0.5,
			mode="gaussian",
			progress=False,
			device=torch.device('cuda')
		)

		output_one = output_one.sigmoid()
		output_two = output_two.sigmoid()
		output_three = output_three.sigmoid()
		output_four = output_four.sigmoid()
		output_five = output_five.sigmoid()

		output_lung = output_lung.sigmoid()

		buffer = []

		buffer.append(output_one)
		buffer.append(output_two)
		buffer.append(output_three)
		buffer.append(output_four)
		buffer.append(output_five)

		output_lobes = torch.cat(buffer, dim=1)

		lung = output_lobes.sum(dim=1).squeeze()
		bg_heatmap = 1 - torch.clip(lung, 0, 1)
		output_lobes = torch.cat([bg_heatmap.unsqueeze(0), output_lobes[0]], dim=0)
		output_lobes = output_lobes.unsqueeze(0)

		return output_lobes, output_lung

	def forward_low(self, x):
		output_one, output_two, output_three, output_four, output_five, lung_output = self.model_low(x)

		output_one = output_one.sigmoid()
		output_two = output_two.sigmoid()
		output_three = output_three.sigmoid()
		output_four = output_four.sigmoid()
		output_five = output_five.sigmoid()

		lung_output = lung_output.sigmoid()

		buffer = []

		buffer.append(output_one)
		buffer.append(output_two)
		buffer.append(output_three)
		buffer.append(output_four)
		buffer.append(output_five)

		output_low = torch.cat(buffer, dim=1)

		lung = output_low.sum(dim=1).squeeze()
		bg_heatmap = 1 - torch.clip(lung, 0, 1)
		output_low = torch.cat([bg_heatmap.unsqueeze(0), output_low[0]], dim=0)
		output_low = output_low.unsqueeze(0)

		return output_low, lung_output

	def forward(self, x_high, x):
		output_low, output_low_lung = self.forward_low(x_high)

		y_low_resize = torch.nn.functional.interpolate(output_low.detach(), size=x[0,0].shape, mode='nearest')

		output_lobes, output_lung = self.forward_per_lobe(x, y_low_resize)

		return y_low_resize, output_lobes, output_low_lung, output_lung

	def test_step(self, test_batch):
		x_high, x = test_batch["image_h"],  test_batch["image"]

		output_low, output_lobes, output_low_lung, output_lung = self.forward(x_high, x)

		return output_lobes.cpu(), output_lung.cpu()

	def predict(self, npz_path, image_original_path, output_path, post_processed=True, save_image=False, rebuild=False) -> np.ndarray:

		if (rebuild):
			assert save_image==True, f'Erro: save_image == False'

		if (save_image):
			ckpt_path = os.path.join(TEMP_IMAGES, 'results/outputs')
			os.makedirs(ckpt_path, exist_ok=True)

		sample = get_sample(npz_path)

		ID_image = os.path.basename(image_original_path).replace('.npz','').replace('.nii.gz','').replace('.nii','').replace('.mhd','').replace('.mha','')



		pre_trained_model_lung_path = 'weights/LightningLung_epoch=90-val_loss=0.014_attUnet_template_lr=0.0001_AdamW_focal_loss_kaggle_saida=6.ckpt'

		test_model_lung = LungModule.load_from_checkpoint(pre_trained_model_lung_path, strict=False)

		#lung = test_model_lung.predict_lung(npz_path)
		lung = test_model_lung.predict(sample, ID_image)

		salvaImageRebuilt(lung.squeeze(), image_original_path, rigid_path=None, ID_image=ID_image, msg='lung', output_path=output_path)



		self.eval()
		with torch.no_grad():
			output_lobes, output_lung = self.test_step(sample)

		#print('Shape:', output_lobes.shape, output_lung.shape)

		output_lung = post_processing_lung(output_lung.squeeze().numpy())

		output_lung = torch.from_numpy(output_lung).float()
		output_lung = output_lung.unsqueeze(dim=0).unsqueeze(dim=0)

		image = output_lobes
		#print('Image shape:', image.shape)

		if post_processed:
			#image = pos_processed(image)

			image = mask_to_onehot(image)
			image = np.expand_dims(image, 0)

			for channel in range(1, image.shape[1]):
				image[0, channel] = post_processing_lung(image[0, channel])

			image = torch.from_numpy(image)
			image = image.squeeze().argmax(dim=0).numpy().astype(np.int8)

			image = post_processing_dist_lung(image, lung)

			assert image.min()==0 and image.max()==5, f'MinMax incorretos {image.shape}: {image.min()} e {image.max()}'

			#print(f'Salvando imagem com pós-processamento final: {image.shape} {image.squeeze().shape}')

			salvaImageRebuilt(image.squeeze(), image_original_path, rigid_path=None, ID_image=ID_image, output_path=output_path)

			del image
			del lung

def main(args):

	image_original_path = sys.argv[1]

	if os.path.isfile(image_original_path):
		all_images = [image_original_path]
	elif os.path.isdir(image_original_path):
		all_images = sorted(glob.glob(os.path.join(image_original_path, '*.nii.gz')))
	else:
		all_images = sorted(glob.glob(os.path.join(image_original_path, '*.nii.gz')))

	print(f'Number of images found in the dataset: {len(all_images)}')

	for image_original_path in all_images:
		ID_image = os.path.basename(image_original_path).replace('.npz','').replace('_affine3D','').replace('_rigid3D','').replace('.nii.gz','').replace('.nii','').replace('_label','').replace('.mhd','')
		print(f'Imagem ID: {ID_image}')

		if os.path.exists(os.path.join(TEMP_IMAGES, 'output_convert_cliped_isometric/images', ID_image+'.nii.gz'))==False:

			os.makedirs(TEMP_IMAGES, exist_ok=True)
			os.makedirs(os.path.join(TEMP_IMAGES, 'output_convert_cliped_isometric/images'), exist_ok=True)

			image, label, lung, airway, spacing, shape = unified_img_reading(
																			image_original_path,
																			torch_convert=False,
																			isometric=True,
																			convert_to_onehot=6)

			transform = torchvision.transforms.Compose([CTHUClip(-1024, 600)])
			image = transform((image, None))

			output_image = sitk.GetImageFromArray(image)

			sitk.WriteImage(output_image, os.path.join(TEMP_IMAGES, 'output_convert_cliped_isometric/images', ID_image+'.nii.gz'))
		else:
			print('Isomeric images successfully created!')

	image_path = os.path.join(TEMP_IMAGES, 'output_convert_cliped_isometric/images', ID_image+'.nii.gz')
	image_data = nib.load(image_path).get_fdata()

	npz_path = os.path.join(TEMP_IMAGES, 'npz_without_registration', f'{ID_image}.npz')
	os.makedirs(os.path.join(TEMP_IMAGES, 'npz_without_registration'), exist_ok=True)

	np.savez_compressed(npz_path, image=image_data, ID=ID_image)

	pre_trained_model_path = 'weights/LightningLobes_6_decoders_no_attUnet_template_epoch=39-val_loss=0.249_attUnet_template_lr=0.0001_AdamW_focal_loss_kaggle_saida=6.ckpt'

	test_model = LoberModuleNormal.load_from_checkpoint(pre_trained_model_path, strict=False)

	test_model.predict(npz_path, image_original_path, post_processed=True, save_image=True, rebuild=True)

	return 0

if __name__ == '__main__':
	os.system('cls' if os.name == 'nt' else 'clear')
	sys.exit(main(sys.argv))
