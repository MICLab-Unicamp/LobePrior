#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dataset import CTDataset3DTemplateAirway
from utils.transform3D import get_transform

class DataModule(pl.LightningDataModule):
	'''
	O datamodul e organiza o carregamento de dados
	'''
	def __init__(self, hparams):
		super().__init__()
		self.save_hyperparameters(hparams)

	def setup(self, stage=None):
		'''
		Definição dos datasets de treino validação e teste e das transformadas.
		'''
		try:
			self.hparams.train_transform = get_transform(self.hparams.train_transform_str)
			self.hparams.eval_transform = get_transform(self.hparams.eval_transform_str)

			self.train = CTDataset3DTemplateAirway("train", transforms=None)
			self.val = CTDataset3DTemplateAirway("val", transforms=None)
		except Exception as e:
			print("Empty dataset!")
			sys.exit(1)

		print("Size of training and validation datasets:",len(self.train),len(self.val))

	def train_dataloader(self):
		trainDataloader = DataLoader(self.train, batch_size=self.hparams.batch_size, num_workers=self.hparams.nworkers, shuffle=True)

		sample = next(iter(trainDataloader))
		img_batch = sample['image']
		seg_batch = sample['label']
		print('Train:')
		print(f"\tFeature batch shape (image): {img_batch.shape}")
		print(f"\tFeature batch shape (label): {seg_batch.shape}")
		print(f"\tMin: {img_batch.min()} Max: {img_batch.max()}")
		print(f"\tMin: {seg_batch.min()} Max: {seg_batch.max()}")
		if self.hparams.datatype=='template':
			template = sample['template']
			print(f"\tFeature shape (template): {template.shape}")
			print(f"\tMinMax (template): {template.min()} {template.max()}")

		return trainDataloader

	def val_dataloader(self):
		valDataloader = DataLoader(self.val, batch_size=self.hparams.batch_size, num_workers=self.hparams.nworkers, shuffle=False)

		sample = next(iter(valDataloader))
		img_batch = sample['image']
		seg_batch = sample['label']
		print('Validation:')
		print(f"\tFeature batch shape (image): {img_batch.shape}")
		print(f"\tFeature batch shape (label): {seg_batch.shape}")
		print(f"\tMin: {img_batch.min()} Max: {img_batch.max()}")
		print(f"\tMin: {seg_batch.min()} Max: {seg_batch.max()}")
		if self.hparams.datatype=='template':
			template = sample['template']
			print(f"\tFeature shape (template): {template.shape}")
			print(f"\tMinMax (template): {template.min()} {template.max()}")

		return valDataloader
