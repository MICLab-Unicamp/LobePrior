#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import torch
import numpy as np
import torchio as tio
import SimpleITK as sitk
import pytorch_lightning as pl
from monai.inferers import sliding_window_inference

from utils.general import post_processing_lung
from model.unet_diedre import UNet_Diedre

HOME = os.getenv("HOME")
RAW_DATA_FOLDER = 'raw_images' #os.path.join(HOME, 'raw_images')

def get_sample_image(npz_path):
	ID_image = os.path.basename(npz_path).replace('.npz','').replace('_affine3D','').replace('_rigid3D','')
	print(f'\tImage name: {ID_image}')

	npz = np.load(npz_path)
	img = npz["image"][:].astype(np.float32)
	print('Shape:', img.shape)
	print('MinMax:', img.min(), img.max())

	group = npz["group"]
	print('Group:', group)
	npz_template_path = os.path.join(RAW_DATA_FOLDER, 'model_fusion/group_'+str(group)+'.npz')

	template = np.load(npz_template_path)["model"][:].astype(np.float32)

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

	return {"image_h": img_high, "image": img, "template": template, "ID_image": ID_image}

class LungModule(pl.LightningModule):
	def __init__(self, hparams):
		super().__init__()

		# O nome precisar ser hparams para o PL.
		self.save_hyperparameters(hparams)

		# Métricas
		#self.metric = Dice_chavg_per_label_metric()
		#self.dice_metric = DiceMetric(include_background=False, reduction="mean")

		if self.hparams.mode == "segmentation":
			self.model_low = UNet_Diedre(n_channels=1, n_classes=1, norm="instance", dim='3d', init_channel=16, joany_conv=False, dict_return=False)
			self.model = UNet_Diedre(n_channels=2, n_classes=1, norm="instance", dim='3d', init_channel=16, joany_conv=False, dict_return=False)

	def forward_per_lobe(self, x, y_seg_resize):

		x_new = torch.cat((x, y_seg_resize), dim = 1)

		output_lung = sliding_window_inference(
			x_new.cuda(),
			roi_size=(128, 128, 128),
			sw_batch_size=1,
			predictor=self.model.cuda(),
			#overlap=0.5,
			mode="gaussian",
			progress=False,
			device=torch.device('cuda')
		)

		output_lung = output_lung.sigmoid()

		return output_lung

	def forward_low(self, x):
		output_low_lung = self.model_low(x)

		output_low_lung = output_low_lung.sigmoid()

		return output_low_lung

	def forward(self, x_high, x):
		output_low = self.forward_low(x_high)

		y_low_resize = torch.nn.functional.interpolate(output_low.detach(), size=x[0,0].shape, mode='nearest')

		output_lung = self.forward_per_lobe(x, y_low_resize)

		return y_low_resize, output_lung

	def test_step(self, val_batch):
		x_high, x = val_batch["image_h"],  val_batch["image"]

		output_low, output_lung = self.forward(x_high, x)

		return output_lung

	def predict_lung(self, npz_path) -> np.ndarray:

		sample = get_sample_image(npz_path)

		self.eval()
		with torch.no_grad():
			output_lung = self.test_step(sample).cpu()

		if isinstance(output_lung, np.ndarray):
			output_lung = torch.from_numpy(output_lung)

		if torch.is_tensor(output_lung):
			output_lung = output_lung.squeeze().squeeze()
			output_lung = output_lung.numpy()

		output_lung = post_processing_lung(output_lung, largest=2)

		#ID_image = os.path.basename(npz_path).replace('.npz','')

		#output = sitk.GetImageFromArray(output_lung.squeeze())
		#sitk.WriteImage(output, os.path.join('', ID_image+"_lung.nii.gz"))

		return output_lung

	def predict(self, sample, ID_image) -> np.ndarray:

		self.eval()
		with torch.no_grad():
			output_lung = self.test_step(sample).cpu()

		if isinstance(output_lung, np.ndarray):
			output_lung = torch.from_numpy(output_lung)

		if torch.is_tensor(output_lung):
			output_lung = output_lung.squeeze().squeeze()
			output_lung = output_lung.numpy()

		output_lung = post_processing_lung(output_lung, largest=2)

		#output = sitk.GetImageFromArray(output_lung.squeeze())
		#sitk.WriteImage(output, os.path.join('', ID_image+"_lung.nii.gz"))

		return output_lung


def main(args):
	print('Parameters:', args)

	npz_path = '/mnt/data/registered_images_no_dice_hcu/test/group_9/460068.npz'
	npz_path = '/mnt/data/temp_images/registered_images/groups/group_9/npz_rigid/460068.npz'

	ID_image = os.path.basename(npz_path).replace('.npz','').replace('.nii.gz','')

	sample = get_sample_image(npz_path)

	template, ID_image = sample['template'], sample['ID_image']

	pre_trained_model_lung_path = '/mnt/data/logs_lung2/LightningLung_epoch=90-val_loss=0.014_attUnet_template_lr=0.0001_AdamW_focal_loss_kaggle_saida=6.ckpt'

	test_model_lung = LungModule.load_from_checkpoint(pre_trained_model_lung_path, strict=False)

	#lung = test_model_lung.predict_lung(npz_path)
	lung = test_model_lung.predict(sample, ID_image)

	#salvaImageRebuilt(lung.squeeze(), image_original_path, rigid_path, ID_image)
	output = sitk.GetImageFromArray(lung.squeeze())
	sitk.WriteImage(output, os.path.join('', ID_image+"_lung.nii.gz"))

	image = sample["image"]
	output = sitk.GetImageFromArray(image.cpu().squeeze())
	sitk.WriteImage(output, os.path.join('', ID_image+"_image2.nii.gz"))

if __name__ == '__main__':
	os.system('cls' if os.name == 'nt' else 'clear')
	sys.exit(main(sys.argv))
