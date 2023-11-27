#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import torch
import random
import threading
import numpy as np
import torchio as tio
from torch.nn import functional as F
from torchvision.transforms import Compose
from torchvision.transforms.transforms import InterpolationMode
from torchvision.transforms import RandomAffine
from typing import Optional, Union, Tuple

class ConditionalCropOrPad():
	'''
	Dynamically crops to 512x512 axial slices or to avoid less than 128 number of slices
	'''
	def __init__(self, min_shape):
		self.min_shape = min_shape

	def __call__(self, x, y):
		with torch.no_grad():
			_, W, H, D = x.shape
			N = y.shape[0]

			# If axial slice dimension is different then target and number of slices W is less, crop
			low_resolution = W < self.min_shape[0]
			if low_resolution or H != self.min_shape[1] or D != self.min_shape[2]:
				if low_resolution:
					target_shape = self.min_shape
				else:
					target_shape = (W,) + self.min_shape[1:]

				self.crop_or_pad_zero = tio.CropOrPad(target_shape, 0)
				self.crop_or_pad_one = tio.CropOrPad(target_shape, 1)

				x = self.crop_or_pad_zero(x)
				y_new = []
				for n in range(N):
					if n == 0:
						y_new.append(self.crop_or_pad_one(y[n:n+1]))
					else:
						y_new.append(self.crop_or_pad_zero(y[n:n+1]))
				y = torch.cat(y_new, dim=0)

			return x, y

	def __str__(self):
		return f"ConditionalCropOrPad {self.min_shape}"


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


# Step 0: have intensities on nnunet expected range
class HUNormalize0(nnUNetTransform):
	'''
	Simplified version of CTHUClipNorm without parametrization

	Note that we have observed strangely good raw HU performance.

	Hypothesis:
		1)This might be due to raw hu indirectly doing augmentation in the visualized range.
		2)Presence of some HU values might "hint" the network on the style of annotation of a dataset.

	Nevertheless, we doing hu normalization for nnUNet-like augmentation due to it being essential to some parametrization.
	'''
	def __init__(self, verbose=False):
		raise DeprecationWarning("HUNormalize was not correctly following nnUNet aug strategy")
		super().__init__(p=None, verbose=verbose)
		self.vmin = -1024
		self.vmax = 600

	def transform(self, x):
		return (((torch.clip(x, self.vmin, self.vmax) - self.vmin)/(self.vmax - self.vmin))*2) - 1

	def __str__(self):
		return "Clip into the -1024, 600 range, min max normalize and extend it to -1, 1"
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


#6
class SimulationOfLowResolution6(nnUNetTransform):
	def __init__(self, verbose=False, mask=False):
		self.mask = mask
		super().__init__(p=0.25, verbose=verbose)

	def transform(self, x: torch.Tensor) -> torch.Tensor:
		# channel+3D input, therefore simulate 5D batch with unsqueeze

		# indirect asseertion of x ndims
		if x.ndim == 4:
			_, Z, Y, X = x.shape
			original_shape: Union[Tuple[int, int], Tuple[int, int, int]] = (Z, Y, X)
			mode = "trilinear"
			align_corners: Optional[bool] = True
		else:
			_, Y, X = x.shape
			original_shape = (Y, X)
			mode = "bilinear"
			align_corners = None

		try:
			scale_factor = random.uniform(0.5, 1)
			x_cache = F.interpolate(x.unsqueeze(0), scale_factor=scale_factor, mode="nearest")
			x = F.interpolate(x_cache, size=original_shape, mode=mode, align_corners=align_corners).squeeze(0)
			if self.mask:
				x = (x > 0.5).float()
			assert (x.shape[0] == 1 or x.shape[0] == 3) and len(x.shape) == len(original_shape) + 1, f"Unexpected interpolation result {x_cache.shape}/{x.shape}"
		except Exception as e:
			print(f"SimulationOfLowResolution error:\n{e}\noriginal_shape: {original_shape} scale_factor: {scale_factor}")
			print(f"Returning original image {x.shape}")

		return x

	def __str__(self):
		return f"6: nnUNet SimulationOfLowResolution with 0.25 probability of nearest downsampling in the U(0.5, 1) range and trilinear upsampling back to original size mask: {self.mask}"


# 7
from torchio.transforms import RandomGamma as tioRandomGamma
class GammaAugmentation7(nnUNetTransform):
	def __init__(self, verbose=False):
		print("GammaAugmentation doesn't work well with 2.5D")
		super().__init__(p=0.15, verbose=verbose)
		# self.augmentation = tioRandomGamma(log_gamma=(0.7, 1.5))

	def transform(self, x: torch.Tensor) -> torch.Tensor:
		# Bring -1, 1 to 0, 1 according to documentation
		# "[...] The patch intensities are scaled to a factor of [0, 1] of their respective value range."
		x_min = x.min()
		x_max = x.max()
		x = (x - x_min)/(x_max - x_min)  #  [1]

		#  "[...] With a probability of 0.15, this augmentation is applied with the voxel intensities being inverted prior to transformation"
		if random.random() <= 0.15:
			x = x.max() - x

		gamma = random.uniform(0.7, 1.5)
		x = x**gamma
		# if x.ndim == 4:
		#     x = self.augmentation(x)
		# else:
		#     x = self.augmentation(x.unsqueeze(0)).squeeze(0)  # include "rgb" 2.5D channel in gamma augmentation

		# "[...] The voxel intensities are subsequently scaled back to their original value range."
		x = x*(x_max - x_min) + x_min  # Solve [1] by x on right hand side

		return x

	def __str__(self):
		return "7: nnUNet Gamma Augmentation with probability 0.15 and gamma parameter between 0.7 and 1.5"


# 8
from torchio.transforms import RandomFlip as tioRandomFlip
class Mirroring8(nnUNetTransform):
	def __init__(self, verbose=False):
		super().__init__(p=None, verbose=verbose)
		self.augmentation = tioRandomFlip(axes=(0, 1, 2), flip_probability=0.5)

	def transform(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim == 4:
			return self.augmentation(x)
		else:
			return self.augmentation(x.unsqueeze(1)).squeeze(1)  # avoid flipping the channel dimension on 2D images

	def __str__(self):
		return "8: nnUNet Mirroring with probability 0.5 for each axis"

# EXTRA fix 0s introduced to onehot
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


class TransformsnnUNet():
	'''
	Updated version of nnunet transforms
	'''
	TRHEAD_SAFE_LOCK = threading.Lock()
	def __init__(self, dim='3d', verbose=False):
		self.dim = dim
		transform = Compose([
						Mirroring8(verbose=verbose),
						RotationAndScaling1(interpolation="linear", verbose=verbose, fill=0, degree_3d=15, scale=(0.7, 1.4)),
						GaussianNoise2(verbose=verbose, raw_hu=False),
						GaussianBlur3(verbose=verbose),
						Brightness4(verbose=verbose),  # added back 23 aug due to contrast enhanced CTs being parse targets
						Contrast5(verbose=verbose),  # added back 23 aug due to contrast enhanced CTs being parse targets
						# SimulationOfLowResolution6(verbose=verbose, mask=False),  # WARNING: this is causing misalignment bugs, removed from training August 14. All slices are 512 512 anyway
						# GammaAugmentation7(verbose=verbose), # Don't agree with gamma augmentation, frequently makes things not visible or too weird.
						#DefinitiveHUNorm(clip=True)  # changed 26/07, random clip is hard to justify when we are removing heart window HU, but still want HU as a definitive intensity measurement
		])
		target_transform = Compose([
						Mirroring8(verbose=verbose),
						RotationAndScaling1(interpolation="nearest", verbose=verbose, degree_3d=15, scale=(0.7, 1.4)),
						FixTargetAfterTransform(verbose=verbose),
						#MorphologicalTransform(verbose=verbose, mode=None, findings_only=True),  # deal with uncertainty of GGO annotation, very important
						# SimulationOfLowResolution6(verbose=verbose, mask=True),
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
		#return f'nnUNet transforms {self.dim}\n' + str(self.transform) + '\n' + str(self.croper) + '\n' + str(self.patcher) + '\n'
		return f'nnUNet transforms {self.dim}\n' + str(self.transform) + '\n'

def main(args):
	transforms  = TransformsnnUNet(verbose=True)
	print(transforms)

	image_path = '/home/jean/DataSets/raw/covid19-ct-scans/COVID-19-CT-Seg_8cases/test/coronacases_006.nii.gz'
	label_path = '/home/jean/DataSets/raw/covid19-ct-scans/COVID lung lobe segmentation/Coronacases_006_lobes.nii'

	import SimpleITK as sitk
	image = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
	label = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
	print(image.shape, label.shape)
	image = torch.from_numpy(np.expand_dims(image, 0)).float()
	image = torch.clip(image, -1024, 600)
	MIN = -1024
	MAX = 600
	image = (image - MIN)/(MAX - MIN)
	label = torch.from_numpy(np.expand_dims(label, 0)).float()

	while True:
		new_image, new_label = transforms(image, label)
		print(new_image.shape, new_label.shape)

		#img, tgt = random_crop(img, tgt, 64, 256, 256)

		#img = torch.from_numpy(img).float()
		#tgt = torch.from_numpy(tgt).float()

		# import matplotlib.pyplot as plt
		# f, (plot0, plot1) = plt.subplots(1, 2, figsize = (12, 6))

		# plot0.imshow(image[0,image.shape[1]//2]), plot0.set_axis_off()
		# plot1.imshow(label[1,label.shape[1]//2]), plot1.set_axis_off()
		# plt.show()
		# plt.close()

		from visualize import surface_render_itksnap
		surface_render_itksnap(img=new_image[0].detach().cpu().numpy(), int_tgt=new_label[0].cpu().numpy(), label="Image", block=True)

if __name__ == '__main__':
	os.system('cls' if os.name == 'nt' else 'clear')
	sys.exit(main(sys.argv))

