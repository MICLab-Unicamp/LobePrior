#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import torch
import threading
import numpy as np
from typing import Tuple
from torchvision.transforms import Compose
from torchvision.transforms import RandomAffine
from torchvision.transforms.transforms import InterpolationMode

"""## Transformadas (Data Augmentation ou Aumentação de Dados)
Transformadas são implementadas como classes chamadas pelo Dataset

Note que operações de Data Augmentation acontecem em tempo real durante o treino.
"""

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

class SegmentationTransform():
	'''
	Applies a torchvision transform into image and segmentation target
	'''
	# thread-safe lock. Multiple processes will have their own random instances, but if you use this with multiple threads,
	# you don't want RNG seed rolling race conditions
	THREAD_SAFE_LOCK = threading.Lock()
	def __init__(self, transform, target_transform, random_swap_code=None):
		if random_swap_code is not None:
			raise DeprecationWarning("Not using random swap anymore, 2023 sprint")

		self.transform = transform
		self.target_transform = target_transform
		self.random_swap_code = random_swap_code
		#self.random_swap = get_random_swap(random_swap_code)

	def __call__(self, image, seg_target):
		'''
		Precisa-se fixar a seed para mesma transformada ser aplicada
		tanto na máscara quanto na imagem.
		'''
		SegmentationTransform.THREAD_SAFE_LOCK.acquire()

		# Gerar uma seed aleatória
		seed = random.randint(0, 2147483647)

		# Fixar seed e aplicar transformada na imagem
		if self.transform is not None:
			random.seed(seed)
			torch.manual_seed(seed)
			np.random.seed(seed)
			image = self.transform(image)

		# Fixar seed e aplicar transformada na máscara
		if self.target_transform is not None:
			random.seed(seed)
			torch.manual_seed(seed)
			np.random.seed(seed)
			seg_target = self.target_transform(seg_target)

		#if self.random_swap is not None:
		#	image, seg_target = self.random_swap(image, seg_target)

		SegmentationTransform.THREAD_SAFE_LOCK.release()
		return image, seg_target

	def __str__(self):
		return f"Image Transform: {self.transform}\n  Target Transform: {self.target_transform}"

class nnUNetTransform():
	'''
	Defined in Page 35 of nnunet published supplementary file.

	https://static-content.springer.com/esm/art%3A10.1038%2Fs41592-020-01008-z/MediaObjects/41592_2020_1008_MOESM1_ESM.pdf

	Number after class name represent the item in page 35 and order of application.
	'''
	def __init__(self, p, verbose=False):
		'''
		p: probability of applying transform. If None, will always call transform.
		'''
		self.verbose = verbose
		self.p = p
		_ = str(self)

	def transform(self, x: torch.Tensor) -> torch.Tensor:
		raise NotImplementedError

	def __call__(self, x: torch.Tensor) -> torch.Tensor:
		with torch.no_grad():
			if self.p is None:
				if self.verbose:
					print(f"Calling transform {self.__class__} because p is None. There might be internal randomization.\n")
				return self.transform(x)
			else:
				p = random.random()
				if p <= self.p:
					if self.verbose:
						print(f"Calling transform {self.__class__} because {p} < {self.p}\n")
					return self.transform(x)
				else:
					if self.verbose:
						print(f"NOT Calling transform {self.__class__} because {p} > {self.p}\n")
					return x

	def __str__(self):
		raise NotImplementedError("Please define __str__ in nnUNetTransforms")

# 1
from torchio.transforms import RandomAffine as tioRandomAffine
class RotationAndScaling1(nnUNetTransform):
	def __init__(self, interpolation, verbose=False, fill=0, degree_2d=180, degree_3d=15, scale=(0.7, 1.4)):
		self.interpolation = interpolation
		self.scale = scale
		self.degree_2d = degree_2d
		self.degree_3d = degree_3d
		self.fill = fill
		super().__init__(p=0.2, verbose=verbose)  # slight deviation from nnunet definition for optimization
		self.augmentation = tioRandomAffine(scales=scale,
											degrees=(-degree_3d, degree_3d),  # ansiotropic range, higher creates too much background
											image_interpolation=interpolation,
											default_pad_value=fill,
											isotropic=True)  # not sure about this but makes sense with "zoom out/in" terminology

		interpolation_2d = {"linear": InterpolationMode.BILINEAR, "nearest": InterpolationMode.NEAREST}
		self.augmentation2d = RandomAffine(degrees=(-degree_2d, degree_2d), translate=None, scale=scale, interpolation=interpolation_2d[interpolation], fill=fill)

	def transform(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim == 4:
			return self.augmentation(x)
		else:
			return self.augmentation2d(x)

	def __str__(self):
		return f"1: RotationAndScaling with 0.2 application probability and scale sampling from {self.scale} and degrees 2D {self.degree_2d}/3D {self.degree_3d}. Interpolation: {self.interpolation}. Fill {self.fill}"

# 2
from torchio.transforms import RandomNoise as tioRandomNoise
class GaussianNoise2(nnUNetTransform):
	def __init__(self, verbose=False, raw_hu=False):
		self.raw_hu = raw_hu
		super().__init__(p=0.15, verbose=verbose)  # nnUNet
		# super().__init__(p=1, verbose=verbose)  # nnUNet
		if self.raw_hu:
			self.augmentation = tioRandomNoise(mean=0, std=(0, 100))
		else:
			self.augmentation = tioRandomNoise(mean=0, std=(0, 0.1))

	def transform(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim == 4:
			return self.augmentation(x)
		else:
			return self.augmentation(x.unsqueeze(1)).squeeze(1)

	def __str__(self):
		return f"2: nnUNet Gaussian Noise 0.15 probability with mean 0 and U(0, 0.1) *1000 if {self.raw_hu} standard deviation"


# 3
from torchio.transforms import RandomBlur as tioRandomBlur
class GaussianBlur3(nnUNetTransform):
	def __init__(self, verbose=False):
		super().__init__(p=0.2, verbose=verbose)
		self.augmentation = tioRandomBlur((0.5, 1.5))

	def transform(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim == 4:
			return self.augmentation(x)
		else:
			return self.augmentation(x.unsqueeze(1)).squeeze(1)

	def __str__(self):
		return "3: nnUNet Gaussian Blur 0.2 probability with mean 0 and U(0.5, 1.5) kernel standard deviation"


# 4
class Brightness4(nnUNetTransform):
	def __init__(self, verbose=False):
		super().__init__(p=0.15, verbose=verbose)

	def transform(self, x: torch.Tensor) -> torch.Tensor:
		return random.uniform(0.7, 1.3)*x

	def __str__(self):
		return "4: nnUNet brightness 0.15 probability with U(0.7, 1.3) brightness"


# 5
class Contrast5(nnUNetTransform):
	def __init__(self, verbose=False):
		super().__init__(p=0.15, verbose=verbose)

	def transform(self, x: torch.Tensor) -> torch.Tensor:
		x_min, x_max = x.min(), x.max()
		return torch.clip(random.uniform(0.65, 1.5)*x, min=x_min, max=x_max)

	def __str__(self):
		return "5: nnUNet contrast 0.15 probability with U(0.65, 1.5) factor clipped to previous min, max range"

class FixTargetAfterTransform(nnUNetTransform):
	def __init__(self, verbose=False):
		super().__init__(p=None, verbose=verbose)

	def transform(self, y):
		# Set spatial locations where onehot check failed to be BG.
		# Skip binary masks
		if y.shape[0] > 1:
			projection = y.sum(dim=0).long()  # guarantees zero presence
			y[0, projection == torch.zeros_like(projection)] = 1.0

		return y

	def __str__(self):
		return "Fix no label area that might be added by transformations, setting it to BG"

class TransformsnnUNet():
	'''
	Updated version of nnunet transforms
	'''
	TRHEAD_SAFE_LOCK = threading.Lock()
	def __init__(self, dim='3d', verbose=False):
		self.dim = dim
		transform = Compose([
						RotationAndScaling1(interpolation="linear", verbose=verbose, fill=0, degree_3d=15, scale=(0.7, 1.4)),
						GaussianNoise2(verbose=verbose, raw_hu=False),
						GaussianBlur3(verbose=verbose),
						Brightness4(verbose=verbose),  # added back 23 aug due to contrast enhanced CTs being parse targets
						Contrast5(verbose=verbose),  # added back 23 aug due to contrast enhanced CTs being parse targets
		])
		target_transform = Compose([
						RotationAndScaling1(interpolation="nearest", verbose=verbose, degree_3d=15, scale=(0.7, 1.4)),
						FixTargetAfterTransform(verbose=verbose),
		])

		self.transform = SegmentationTransform(transform=transform, target_transform=target_transform)

	def __call__(self, x, y):
		TransformsnnUNet.TRHEAD_SAFE_LOCK.acquire()  # Avoid RNG race conditions with patcher and nnunet transforms. IO is still parallel.
		if self.dim == '2d':
			assert x.ndim == 3
		elif self.dim == '3d':
			assert x.ndim == 4

		x, y = self.transform(x, y)
		#x, y = self.croper(x, y)
		#x, y = self.patcher(x, y)
		TransformsnnUNet.TRHEAD_SAFE_LOCK.release()
		return x, y

	def __str__(self):
		return f'nnUNet transforms {self.dim}\n' + str(self.transform) + '\n'

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
