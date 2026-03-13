#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np
import nibabel as nib
import torchvision
import torchio as tio
import SimpleITK as sitk
import pytorch_lightning as pl

from monai.inferers import sliding_window_inference

from src.model.unet_diedre import UNet_Diedre
from src.utils.general import (
	post_processing_lung,
	unified_img_reading,
	salvaImageRebuilt,
	convert_to_nifti,
	collect_images_verbose,
)
from src.utils.transform3D import CTHUClip


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
HOME = os.getenv("HOME")
TEMP_IMAGES = "temp_images"


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def get_sample_image(npz_path: str):
	"""Load NPZ image and prepare TorchIO tensors."""
	ID_image = os.path.basename(npz_path).replace(".npz", "")
	print(f"\tImage name: {ID_image}")

	npz = np.load(npz_path)
	img = npz["image"].astype(np.float32)

	print("Shape:", img.shape)
	print("MinMax:", img.min(), img.max())

	# (Z, Y, X) -> (X, Y, Z)
	img = img.transpose(2, 1, 0)

	if img.ndim == 3:
		img = np.expand_dims(img, axis=0)

	subject = tio.Subject(
		image=tio.ScalarImage(tensor=img),
	)

	transform = tio.Resize((128, 128, 128))
	transformed = transform(subject)
	img_high = transformed.image.numpy()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	img_high = torch.tensor(img_high, dtype=torch.float32).unsqueeze(0).to(device)
	img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)

	return {"image_h": img_high, "image": img}


# -----------------------------------------------------------------------------
# Lightning Module
# -----------------------------------------------------------------------------
class LungModule(pl.LightningModule):
	def __init__(self, hparams=None):
		super().__init__()

		# FIX: hparams pode ser None ao carregar checkpoint
		if hparams is None:
			hparams = argparse.Namespace(mode="segmentation")

		self.save_hyperparameters(hparams)

		if self.hparams.mode == "segmentation":
			self.model_low = UNet_Diedre(
				n_channels=1,
				n_classes=1,
				norm="instance",
				dim="3d",
				init_channel=16,
				dict_return=False,
			)

			self.model = UNet_Diedre(
				n_channels=2,
				n_classes=1,
				norm="instance",
				dim="3d",
				init_channel=16,
				dict_return=False,
			)

	def forward_low(self, x):
		out = self.model_low(x)
		return out.sigmoid()

	def forward_per_lobe(self, x, y_seg_resize):
		x_new = torch.cat((x, y_seg_resize), dim=1)

		out = sliding_window_inference(
			x_new,
			roi_size=(128, 128, 128),
			sw_batch_size=1,
			predictor=self.model,
			mode="gaussian",
			progress=False,
		)

		return out.sigmoid()

	def forward(self, x_high, x):
		output_low = self.forward_low(x_high)

		# FIX: interpolate correto
		y_low_resize = torch.nn.functional.interpolate(
			output_low.detach(),
			size=x.shape[2:],
			mode="nearest",
		)

		output_lung = self.forward_per_lobe(x, y_low_resize)
		return y_low_resize, output_lung

	@torch.no_grad()
	def test_step(self, batch):
		x_high, x = batch["image_h"], batch["image"]
		_, output_lung = self.forward(x_high, x)
		return output_lung

	@torch.no_grad()
	def predict_lung(self, npz_path: str) -> np.ndarray:
		sample = get_sample_image(npz_path)

		self.eval()
		output_lung = self.test_step(sample)

		if torch.is_tensor(output_lung):
			output_lung = output_lung.squeeze().cpu().numpy()

		output_lung = post_processing_lung(output_lung, largest=2)
		return output_lung


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------
def run(args):
	print("Parameters:", args)

	image_original_path = args.input
	output_path = args.output
	delete_data = args.delete

	print(f"Input: {image_original_path}")
	print(f"Output: {output_path}")
	print(f"Delete temporary files: {delete_data}")

	all_images = collect_images_verbose(image_original_path)

	if len(all_images) == 0:
		print("No input images found.")
		return 0

	for image_original_path in all_images:
		path = Path(image_original_path)
		ext = "".join(path.suffixes)

		if ext in [".mhd", ".mha"]:
			image_original_path = convert_to_nifti(image_original_path)

		ID_image = (
			os.path.basename(image_original_path)
			.replace(".nii.gz", "")
			.replace(".nii", "")
		)

		print(f"Image ID: {ID_image}")

		iso_dir = os.path.join(
			TEMP_IMAGES, "output_convert_cliped_isometric", "images"
		)
		os.makedirs(iso_dir, exist_ok=True)

		iso_image_path = os.path.join(iso_dir, f"{ID_image}.nii.gz")

		if not os.path.exists(iso_image_path):
			image, _, _, _, _, _ = unified_img_reading(
				image_original_path,
				torch_convert=False,
				isometric=True,
				convert_to_onehot=6,
			)

			transform = torchvision.transforms.Compose([CTHUClip(-1024, 600)])
			image = transform((image, None))

			sitk.WriteImage(
				sitk.GetImageFromArray(image),
				iso_image_path,
			)
		else:
			print("Isometric image already exists.")

		# ---------------------------------------------------------------------
		# NPZ conversion
		# ---------------------------------------------------------------------
		image_np = nib.load(iso_image_path).get_fdata().astype(np.float32)

		npz_dir = os.path.join(TEMP_IMAGES, "registered_images", "npz_rigid")
		os.makedirs(npz_dir, exist_ok=True)

		npz_path = os.path.join(npz_dir, f"{ID_image}.npz")
		np.savez_compressed(npz_path, image=image_np)

		# ---------------------------------------------------------------------
		# Load model
		# ---------------------------------------------------------------------
		ckpt_path = "src/weights/LightningLung.ckpt"

		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		model = LungModule.load_from_checkpoint(
			ckpt_path,
			strict=False,
			map_location=device,
		).to(device)

		lung = model.predict_lung(npz_path)

		salvaImageRebuilt(
			lung,
			image_original_path,
			rigid_path=None,
			ID_image=ID_image,
			msg="lung",
			output_path=output_path,
		)

	return 0


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
	parser = argparse.ArgumentParser(
		description="Lung segmentation on CT images using prior information."
	)
	parser.add_argument("-i", "--input", required=True, help="Input image or folder")
	parser.add_argument("-o", "--output", default="outputs", help="Output directory")
	parser.add_argument("-d", "--delete", action="store_true", help="Delete temp files")

	args = parser.parse_args()
	return run(args)


if __name__ == "__main__":
	os.system("cls" if os.name == "nt" else "clear")
	sys.exit(main())
