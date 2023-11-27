#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import torch
import numpy as np
from torchvision import transforms

"""## Transformadas (Data Augmentation ou Aumentação de Dados)
Transformadas são implementadas como classes chamadas pelo Dataset

Note que operações de Data Augmentation acontecem em tempo real durante o treino.
"""

class RandomNoise():
	def __init__(self, mu=0, sigma=0.1):
		self.mu = mu
		self.sigma = sigma

	def __call__(self, image):
		noise = np.clip(self.sigma * np.random.randn(image.shape[1], image.shape[2], image.shape[3]), -2*self.sigma, 2*self.sigma)
		noise = noise + self.mu
		image = image + noise
		return image

class RandomBrightness():
	def __init__(self):
		self.max = 255
		self.min = 0

	def __call__(self, image):
		c = np.random.randint(-20, 20)

		image = image + c

		image[image >= self.max] = self.max
		image[image <= self.min] = self.min

		return image

class RandomContrast():
	def __init__(self):
		self.c = np.random.randint(-20, 20)

	def __call__(self, image):
		shape = image.shape
		ntotpixel = shape[0] * shape[1] * shape[2]
		IOD = np.sum(image)
		luminanza = int(IOD / ntotpixel)

		d = image - luminanza
		dc = d * abs(self.c) / 100

		if self.c >= 0:
			J = image + dc
			J[J >= 255] = 255
			J[J <= 0] = 0
		else:
			J = image - dc
			J[J >= 255] = 255
			J[J <= 0] = 0

		return J

class SegmentationTransform():
	'''
	Applies a torchvision transform into image and segmentation target
	'''
	def __init__(self, transform, target_transform):
		self.transform = transform
		self.target_transform = target_transform

	def __call__(self, image, mask):
		'''
		Precisa-se fixar a seed para mesma transformada ser aplicada tanto na máscara quanto na imagem.
		'''
		# Gerar uma seed aleatória
		seed = np.random.randint(2147483647)

		# Fixar seed e aplicar transformada na imagem
		random.seed(seed)
		torch.manual_seed(seed)
		#if self.transform is not None:
		#    transformed = self.transform(image=image, mask=mask)
		#    image = transformed["image"]
		#    mask = transformed["mask"]

		random.seed(seed)
		torch.manual_seed(seed)
		if self.transform is not None:
			image = self.transform(image)
		random.seed(seed)
		torch.manual_seed(seed)
		if self.target_transform is not None:
			mask = self.target_transform(mask)

		return image, mask

def get_transform(string):
	print('transform:', string)
	'''
	Mapeamento de uma string a uma transformadas.
	'''
	if string is None:
		image_transform = None
		target_transform = None
	elif string == "my_transforms":
		image_transform = transforms.Compose([
											RandomNoise(),
											RandomBrightness(),
											RandomContrast(),
											])
		target_transform = transforms.Compose([

											])
	else:
		raise ValueError(f"{string} does not correspond to a transform.")

	return SegmentationTransform(image_transform, target_transform)

def random_crop(image, label, depth, width, height):
	if len(image.shape)!=4 and len(label.shape)!=4:
		print(f'Tamanho de shape diferente de 4.')

	assert (len(image.shape)==4 and len(label.shape)==4)
	assert image.shape[2] == image.shape[3]

	if (image.shape[1] < depth):
		depth = image.shape[1]
	assert image.shape[1] >= depth
	assert image.shape[2] >= height
	assert image.shape[3] >= width

	z = random.randint(0, image.shape[1] - depth)
	x = random.randint(0, image.shape[2] - height)
	y = random.randint(0, image.shape[3] - width)

	#print(image.shape[1], depth)
	#print(image.shape[2], height)
	#print(image.shape[3], width)

	image = image[:, z:z+depth, x:x+height, y:y+width]
	label = label[:, z:z+depth, x:x+height, y:y+width]

	return image, label
