#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import random
import monai
import pytorch_lightning as pl
from monai.inferers import sliding_window_inference

from model.unet_diedre import UNet_SeteDecoders
from utils.metric import Dice_chavg_per_label_metric
from utils.transform3D import random_crop

class Lightning(pl.LightningModule):
	def __init__(self, hparams):
		super().__init__()

		# O nome precisar ser hparams para o PL.
		self.save_hyperparameters(hparams)

		# Loss
		#self.criterion = FocalLoss()
		self.criterion = monai.losses.DiceLoss(reduction='mean')

		# Métricas
		self.metric = Dice_chavg_per_label_metric()

		if self.hparams.mode == "segmentation":
			# Hat: opera em low resolution (fov inteiro)
			self.model_low = UNet_SeteDecoders(n_channels=1, n_classes=1, norm="instance", dim='3d', init_channel=16, joany_conv=False, dict_return=False)
			# Seg: opera em high resolution (patch)
			self.model = UNet_SeteDecoders(n_channels=14, n_classes=1, norm="instance", dim='3d', init_channel=16, joany_conv=False, dict_return=False)

	def forward_per_lobe(self, x, template, y_seg_resize, y_lung, y_lobe, y_airway):

		#template = (template > 0.3).float()

		x_new = torch.cat((x, y_seg_resize, template), dim = 1)

		if self.training:
			x_new, y_lung_new, y_lobe_new, y_airway_new = random_crop(x_new, y_lung, y_lobe, y_airway, crop_size=(96, 96, 96))

			output_lung, output_one, output_two, output_three, output_four, output_five, output_airway = self.model(x_new)
		else:
			y_lung_new = y_lung
			y_lobe_new = y_lobe
			y_airway_new = y_airway

			output_lung, output_one, output_two, output_three, output_four, output_five, output_airway = sliding_window_inference(
				x_new.cuda(),
				roi_size=(96, 96, 96),
				sw_batch_size=1,
				predictor=self.model.cuda(),
				#overlap=0.5,
				mode="gaussian",
				progress=False,
				device=torch.device('cuda')
			)

		output_lung = output_lung.sigmoid()

		output_lung = 1-output_lung

		output_one = output_one.sigmoid()
		output_two = output_two.sigmoid()
		output_three = output_three.sigmoid()
		output_four = output_four.sigmoid()
		output_five = output_five.sigmoid()

		output_airway = output_airway.sigmoid()

		buffer = []

		buffer.append(output_lung)
		buffer.append(output_one)
		buffer.append(output_two)
		buffer.append(output_three)
		buffer.append(output_four)
		buffer.append(output_five)

		output_lobes = torch.cat(buffer, dim=1)

		loss_lung = self.criterion(output_lung, y_lobe_new[:, 0:1])

		loss_one = self.criterion(output_one, y_lobe_new[:, 1:2])
		loss_two = self.criterion(output_two, y_lobe_new[:, 2:3])
		loss_three = self.criterion(output_three, y_lobe_new[:, 3:4])
		loss_four = self.criterion(output_four, y_lobe_new[:, 4:5])
		loss_five = self.criterion(output_five, y_lobe_new[:, 5:6])

		loss_airway = self.criterion(output_airway, y_airway_new)

		loss_lobes = loss_lung+loss_one+loss_two+loss_three+loss_four+loss_five

		if self.training:
			return loss_lobes, loss_airway
		else:
			return output_lobes, output_airway, loss_lobes, loss_airway

	def forward_low(self, x, lobes_high, airway_high):
		output_lung, output_one, output_two, output_three, output_four, output_five, airway = self.model_low(x)

		output_lung = output_lung.sigmoid()

		output_lung = 1-output_lung

		output_one = output_one.sigmoid()
		output_two = output_two.sigmoid()
		output_three = output_three.sigmoid()
		output_four = output_four.sigmoid()
		output_five = output_five.sigmoid()

		airway = airway.sigmoid()

		buffer = []

		buffer.append(output_lung)
		buffer.append(output_one)
		buffer.append(output_two)
		buffer.append(output_three)
		buffer.append(output_four)
		buffer.append(output_five)
		buffer.append(airway)

		output_low = torch.cat(buffer, dim=1)

		loss_lung = self.criterion(output_lung, lobes_high[:, 0:1])

		loss_one = self.criterion(output_one, lobes_high[:, 1:2])
		loss_two = self.criterion(output_two, lobes_high[:, 2:3])
		loss_three = self.criterion(output_three, lobes_high[:, 3:4])
		loss_four = self.criterion(output_four, lobes_high[:, 4:5])
		loss_five = self.criterion(output_five, lobes_high[:, 5:6])

		loss_airway = self.criterion(airway, airway_high)

		loss_lobes = loss_lung+loss_one+loss_two+loss_three+loss_four+loss_five

		return output_low, loss_lobes+loss_airway

	def forward(self, x_high, y_high, airway_high, x, y_lobe, y_airway, y_lung, template):
		output_low, loss_low_lobes = self.forward_low(x_high, y_high, airway_high)

		y_low_resize = torch.nn.functional.interpolate(output_low.detach(), size=x[0,0].shape, mode='nearest')

		if self.training:
			loss_lobes, loss_airway = self.forward_per_lobe(x, template, y_low_resize, y_lung, y_lobe, y_airway)
			return loss_low_lobes, loss_lobes, loss_airway
		else:
			output_lobes, output_airway, loss_lobes, loss_airway = self.forward_per_lobe(x, template, y_low_resize, y_lung, y_lobe, y_airway)
			return output_low, output_lobes, output_airway, loss_lobes, loss_airway

	def training_step(self, train_batch, batch_idx):
		x_high, y_lobes_high, y_airway_high, x, y_lobes, y_airway, template = train_batch["image_h"], train_batch["label_h"], train_batch["airway_h"], train_batch["image"], train_batch["label"], train_batch["airway"], train_batch["template"]

		y_lung = y_lobes.clone()
		y_lung = y_lung.squeeze().argmax(dim=0)
		y_lung[y_lung>0]=1
		y_lung = y_lung.unsqueeze(dim=0).unsqueeze(dim=0)

		loss_low, loss_lobes, loss_airway = self.forward(x_high, y_lobes_high, y_airway_high, x, y_lobes, y_airway, y_lung, template)

		self.log("train_loss_low", loss_low, on_epoch=True, on_step=True, batch_size=self.hparams.batch_size)
		self.log("train_loss_lobes", loss_lobes, on_epoch=True, on_step=True, batch_size=self.hparams.batch_size)
		self.log("train_loss_airway", loss_airway, on_epoch=True, on_step=True, batch_size=self.hparams.batch_size)

		return loss_low+loss_lobes+loss_airway

	def validation_step(self, val_batch, batch_idx):
		x_high, y_lobes_high, y_airway_high, x, y_lobes, y_airway, template = val_batch["image_h"], val_batch["label_h"], val_batch["airway_h"], val_batch["image"], val_batch["label"], val_batch["airway"], val_batch["template"]

		y_lung = y_lobes.clone()
		y_lung = y_lung.squeeze().argmax(dim=0)
		y_lung[y_lung>0]=1
		y_lung = y_lung.unsqueeze(dim=0).unsqueeze(dim=0)

		output_low, output_lobes, output_airway, loss_lobes, loss_airway = self.forward(x_high, y_lobes_high, y_airway_high, x, y_lobes, y_airway, y_lung, template)

		dsc_low_lobes = self.metric(output_low[:,0:6], y_lobes_high)[0]
		dsc_low_airway = self.metric(output_low[:,6:7], y_airway_high)[0]

		dsc_mean, dsc_lobes = self.metric(output_lobes[:,0:6], y_lobes)
		dsc_airway = self.metric(output_airway, y_airway)[0]

		batch_size = self.hparams.batch_size

		# Losses
		self.log("val_loss_lobes", loss_lobes, on_epoch=True, on_step=True, batch_size=self.hparams.batch_size)
		self.log("val_loss_airway", loss_lobes+loss_airway, on_epoch=True, on_step=True, batch_size=self.hparams.batch_size)

		# Dice mean
		self.log("val_lobes_low", dsc_low_lobes.cpu(), on_epoch=True, on_step=False, prog_bar=True, batch_size=batch_size)
		self.log("val_airway_low", dsc_low_airway.cpu(), on_epoch=True, on_step=False, prog_bar=True, batch_size=batch_size)

		self.log("val_lobes", dsc_mean, on_epoch=True, on_step=False, prog_bar=True, batch_size=batch_size)
		self.log("val_airway", dsc_airway, on_epoch=True, on_step=False, prog_bar=True, batch_size=batch_size)

		# Dice by lobes
		self.log("val_dsc_bg", dsc_lobes[0].cpu(), on_epoch=True, on_step=False, prog_bar=True, batch_size=batch_size)
		self.log("val_dsc_lul", dsc_lobes[1].cpu(), on_epoch=True, on_step=False, prog_bar=True, batch_size=batch_size)
		self.log("val_dsc_lll", dsc_lobes[2].cpu(), on_epoch=True, on_step=False, prog_bar=True, batch_size=batch_size)
		self.log("val_dsc_rul", dsc_lobes[3].cpu(), on_epoch=True, on_step=False, prog_bar=True, batch_size=batch_size)
		self.log("val_dsc_rml", dsc_lobes[4].cpu(), on_epoch=True, on_step=False, prog_bar=True, batch_size=batch_size)
		self.log("val_dsc_rll", dsc_lobes[5].cpu(), on_epoch=True, on_step=False, prog_bar=True, batch_size=batch_size)

	def configure_optimizers(self):
		optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-5)

		if self.hparams.scheduler is None:
			print('Utilizando otimizador {} e lr {} com weight_decay {} sem lr_scheduler.'.format(optimizer, self.hparams.lr, self.hparams.weight_decay))
			return optimizer
		else:
			if (self.hparams.scheduler=='StepLR'):
				scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.985)
			elif (self.hparams.scheduler=='CosineAnnealingLR'):
				scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
			else:
				print(f'Scheduler não encontrado: {self.hparams.lr_scheduler}')
			print(f'Utilizando {optimizer} com scheduler {self.hparams.scheduler}.')

			return [optimizer], [scheduler]
