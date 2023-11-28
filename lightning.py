#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import monai
import skimage
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from monai.inferers import sliding_window_inference

from model.unet_diedre import UNet_Diedre
from utils.loss import BayesianLossByChannel
from utils.metric import Dice_metric, Dice_chavg_per_label_metric
from utils.transform3D import random_crop

class Lightning(pl.LightningModule):
	def __init__(self, hparams):
		super().__init__()

		# O nome precisar ser hparams para o PL.
		self.save_hyperparameters(hparams)

		# Loss
		self.criterion_baysian = BayesianLossByChannel()
		self.criterion_by_lobe = monai.losses.DiceLoss(reduction='mean')

		# Métricas
		self.metric = Dice_chavg_per_label_metric()

		self.automatic_optimization = False

		if self.hparams.mode == "segmentation":
			self.model_hat = UNet_Diedre(n_channels=1, n_classes=6, norm="instance", dim='3d', init_channel=16, joany_conv=False, dict_return=False)
			self.model_seg_0 = UNet_Diedre(n_channels=3, n_classes=1, norm="instance", dim='3d', init_channel=8, joany_conv=False, dict_return=False)
			self.model_seg_1 = UNet_Diedre(n_channels=3, n_classes=1, norm="instance", dim='3d', init_channel=8, joany_conv=False, dict_return=False)
			self.model_seg_2 = UNet_Diedre(n_channels=3, n_classes=1, norm="instance", dim='3d', init_channel=8, joany_conv=False, dict_return=False)
			self.model_seg_3 = UNet_Diedre(n_channels=3, n_classes=1, norm="instance", dim='3d', init_channel=8, joany_conv=False, dict_return=False)
			self.model_seg_4 = UNet_Diedre(n_channels=3, n_classes=1, norm="instance", dim='3d', init_channel=8, joany_conv=False, dict_return=False)

	def forward_per_lobe(self, x, template, y_seg, y):

		template = (template > 0.3).float()

		lobes_regions = skimage.measure.regionprops(np.array(template.squeeze().cpu().argmax(axis=0)).astype(np.uint8))

		loss_cor = []
		dice_cor = []

		for i in range(5):
			lobe_region = lobes_regions[i].bbox

			# Make lobe region a 1 cube
			box = np.zeros((1, 1) + template[0,0].shape, dtype=np.uint8)
			box[:,:,lobe_region[0]:lobe_region[3], lobe_region[1]:lobe_region[4], lobe_region[2]:lobe_region[5]] = 1

			label_seg = y_seg[:,i+1:i+2]

			# Concatenate cube with CT
			box = torch.from_numpy(box).cuda()

			#print(x.shape, label_seg.shape, box.shape)
			x_new = torch.cat((x, label_seg, box), dim = 1)

			if self.training:
				x_new, y_new = random_crop(x_new.squeeze(0), y.squeeze(0), 128, 256, 256)
				x_new, y_new = x_new.unsqueeze(0), y_new.unsqueeze(0)

				if i==0:
					output = self.model_seg_0(x_new)
				elif i==1:
					output = self.model_seg_1(x_new)
				elif i==2:
					output = self.model_seg_2(x_new)
				elif i==3:
					output = self.model_seg_3(x_new)
				elif i==4:
					output = self.model_seg_4(x_new)
			else:
				if i==0:
					model = self.model_seg_0
				elif i==1:
					model = self.model_seg_1
				elif i==2:
					model = self.model_seg_2
				elif i==3:
					model = self.model_seg_3
				elif i==4:
					model = self.model_seg_4
				y_new = y

				output = sliding_window_inference(
					x_new.cuda(),
					roi_size=(128, 256, 256),
					sw_batch_size=1,
					predictor=model.cuda(),
					overlap=0.5,mode="gaussian",
					progress=False,
					device=torch.device('cuda')
				)

			output = output.sigmoid()

			local_loss = self.criterion_by_lobe(output, y_new[:, i+1:i+2])
			loss_cor.append(local_loss)
			if self.training:
				self.manual_backward(local_loss)
			else:
				dice_cor.append(Dice_metric(output, y_new[:,i+1:i+2]).mean())

		# Concatenate on channel dimension
		#y_hat_seg = torch.cat(buffer, dim=1)

		if self.training==False:
			return dice_cor, loss_cor
		else:
			return loss_cor

	def forward_seg(self, x):
		y_hat = self.model_hat(x)

		return y_hat.softmax(dim=1)

	def forward(self, x_high, x, template, y):
		y_seg = self.forward_seg(x_high)
		y_seg_resize = torch.nn.functional.interpolate(y_seg.detach(), size=x[0,0].shape, mode='nearest')

		if self.training:
			loss_cor = self.forward_per_lobe(x, template, y_seg_resize, y)
			return y_seg, loss_cor
		else:
			dice_cor, loss_cor = self.forward_per_lobe(x, template, y_seg_resize, y)
			return y_seg, dice_cor, loss_cor

	def training_step(self, train_batch, batch_idx):
		opt = self.optimizers()
		opt.zero_grad()

		x_high, y_high, template_high, x, y, template = train_batch["image_h"], train_batch["label_h"], train_batch["template_high"], train_batch["image"], train_batch["label"], train_batch["template"]
		y_seg, loss_cor = self.forward(x_high, x, template, y)

		assert y_seg.shape==y_high.shape
		loss_seg = self.criterion_baysian(y_seg, y_high, template_high)
		self.manual_backward(loss_seg)

		loss_cor_all = torch.stack(loss_cor).mean()

		opt.step()

		self.log("loss_seg", loss_seg, on_epoch=True, on_step=False, batch_size=self.hparams.batch_size)
		self.log("train_loss",  loss_cor_all, on_epoch=True, on_step=True, batch_size=self.hparams.batch_size)

	def validation_step(self, val_batch, batch_idx):
		x_high, y_high, template_high, x, y, template = val_batch["image_h"], val_batch["label_h"], val_batch["template_high"], val_batch["image"], val_batch["label"], val_batch["template"]
		y_seg, dice_cor, loss_cor = self.forward(x_high, x, template, y)

		assert y_seg.shape==y_high.shape
		loss_seg = self.criterion_baysian(y_seg, y_high, template_high)
		dsc_seg, dsc_lobes_seg = self.metric(y_seg, y_high)

		loss_cor_all = torch.stack(loss_cor).mean()

		batch_size = self.hparams.batch_size

		# Loss rede inicial grande
		self.log("val_loss_seg", loss_seg, on_epoch=True, on_step=False, prog_bar=False, batch_size=batch_size)
		self.log("val_loss", loss_cor_all, on_epoch=True, on_step=False, prog_bar=False, batch_size=batch_size)

		# Dice rede inicial grande
		self.log("val_dice_seg", dsc_seg.cpu(), on_epoch=True, on_step=False, prog_bar=True, batch_size=batch_size)
		self.log("val_dsc_lul_seg", dsc_lobes_seg[0].cpu(), on_epoch=True, on_step=False, prog_bar=True, batch_size=batch_size)
		self.log("val_dsc_lll_seg", dsc_lobes_seg[1].cpu(), on_epoch=True, on_step=False, prog_bar=True, batch_size=batch_size)
		self.log("val_dsc_rul_seg", dsc_lobes_seg[2].cpu(), on_epoch=True, on_step=False, prog_bar=True, batch_size=batch_size)
		self.log("val_dsc_rml_seg", dsc_lobes_seg[3].cpu(), on_epoch=True, on_step=False, prog_bar=True, batch_size=batch_size)
		self.log("val_dsc_rll_seg", dsc_lobes_seg[4].cpu(), on_epoch=True, on_step=False, prog_bar=True, batch_size=batch_size)

		# Focal Loss redes pequenas (per lobe)
		self.log("loss_lul", loss_cor[0].cpu(), on_epoch=True, on_step=False, prog_bar=True, batch_size=batch_size)
		self.log("loss_lll", loss_cor[1].cpu(), on_epoch=True, on_step=False, prog_bar=True, batch_size=batch_size)
		self.log("loss_rul", loss_cor[2].cpu(), on_epoch=True, on_step=False, prog_bar=True, batch_size=batch_size)
		self.log("loss_rml", loss_cor[3].cpu(), on_epoch=True, on_step=False, prog_bar=True, batch_size=batch_size)
		self.log("loss_rll", loss_cor[4].cpu(), on_epoch=True, on_step=False, prog_bar=True, batch_size=batch_size)

		# Dice redes pequenas (per lobe)
		#self.log("val_dice", dsc.cpu(), on_epoch=True, on_step=False, prog_bar=True, batch_size=batch_size)
		self.log("val_dsc_lul", dice_cor[0].cpu(), on_epoch=True, on_step=False, prog_bar=True, batch_size=batch_size)
		self.log("val_dsc_lll", dice_cor[1].cpu(), on_epoch=True, on_step=False, prog_bar=True, batch_size=batch_size)
		self.log("val_dsc_rul", dice_cor[2].cpu(), on_epoch=True, on_step=False, prog_bar=True, batch_size=batch_size)
		self.log("val_dsc_rml", dice_cor[3].cpu(), on_epoch=True, on_step=False, prog_bar=True, batch_size=batch_size)
		self.log("val_dsc_rll", dice_cor[4].cpu(), on_epoch=True, on_step=False, prog_bar=True, batch_size=batch_size)

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
