#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import torch
import argparse
import torchvision
import numpy as np
import torchio as tio
import SimpleITK as sitk
import nibabel as nib
import pytorch_lightning as pl
from monai.inferers import sliding_window_inference
from pathlib import Path

from utils.general import analyze_registration_quality, find_best_registration
from utils.general import post_processing_lung
from utils.general import unified_img_reading, busca_path, salvaImageRebuilt
from model.unet_diedre import UNet_Diedre
from utils.transform3D import CTHUClip

HOME = os.getenv("HOME")
TEMP_IMAGES = 'temp_images'
RAW_DATA_FOLDER = 'raw_images' #os.path.join(HOME, 'raw_images')

def get_sample_image(npz_path):
	ID_image = os.path.basename(npz_path).replace('.npz','').replace('_affine3D','').replace('_rigid3D','')
	print(f'\tImage name: {ID_image}')

	npz = np.load(npz_path)
	img = npz["image"][:].astype(np.float32)
	print('Shape:', img.shape)
	print('MinMax:', img.min(), img.max())

	#group = npz["group"]
	#print('Group:', group)
	#npz_template_path = os.path.join(RAW_DATA_FOLDER, 'model_fusion/group_'+str(group)+'.npz')

	#template = np.load(npz_template_path)["model"][:].astype(np.float32)

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

	return {"image_h": img_high, "image": img}

class LungModule(pl.LightningModule):
	def __init__(self, hparams):
		super().__init__()

		# O nome precisar ser hparams para o PL.
		self.save_hyperparameters(hparams)

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

	@torch.no_grad()
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

		return output_lung


def main(args):
	print('Parameters:', args)

	modo_register = True
	delete_data = False
	output_path = os.path.join(TEMP_IMAGES, 'outputs')

	parser = argparse.ArgumentParser(description='Lung lobe segmentation on CT images using prior information.')
	parser.add_argument('--input', "-i", default="inputs", help= "Input image or folder with volumetric images.", type=str)
	parser.add_argument('--output', "-o", default="outputs", help= "Directory to store the final segmentation.", type=str)
	parser.add_argument('--normal', "-n", action="store_true", help= "Use Prior Information.") 			# true se passou --normal
	parser.add_argument('--delete', "-d", action="store_true", help= "Delete temporary files.") 		# true se passou --delete

	args = parser.parse_args()

	image_original_path = args.input
	output_path = args.output
	modo_normal = args.normal
	delete_data = args.delete

	print(f'Input: {image_original_path}')
	print(f'Output: {output_path}')
	print(f'Prior Information: {modo_register}')
	print(f'Delete temporary files : {delete_data}')

	if os.path.isfile(image_original_path):
		path = Path(image_original_path)
		ext = "".join(path.suffixes)
		if ext not in ['.nii', '.nii.gz', '.mhd', '.mha']:
			print(f'The file format is not valid: {ext}')
			print(f'The image name must not contain dots or the image extension must be .nii, .nii.gz, .mhd or .mha')
			return 0
		all_images = [image_original_path]
	elif os.path.isdir(image_original_path):
		extensoes = ['*.nii', '*.nii.gz', '*.mhd', '*.mha']
		all_images = []
		for ext in extensoes:
			all_images.extend(glob.glob(os.path.join(image_original_path, ext)))
	else:
		all_images = sorted(glob.glob(os.path.join(image_original_path, '*.nii.gz')))

	print(f'Number of images found in the dataset: {len(all_images)}')

	for image_original_path in all_images:
		ID_image = os.path.basename(image_original_path).replace('.npz','').replace('_affine3D','').replace('_rigid3D','').replace('.nii.gz','').replace('.nii','').replace('_label','').replace('.mhd','')
		print(f'Image ID: {ID_image}')

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

		N_THREADS = mp.cpu_count()//2
		arg_list = []
		pool = mp.Pool(N_THREADS)

		for group in range(1,11):
			if teste_pickle_by_image(ID_image, group)==False:
				register_single(image_path, None, None, None, group)
		#		arg_list.append((image_path, None, None, None, group))

		#for _ in tqdm(pool.imap_unordered(register_single, arg_list)):
		#	pass

		print('Registration completed successfully!')

		image = nib.load(image_path).get_fdata()

		# Analisa todas as imagens
		registered_folder = os.path.join(RAW_DATA_FOLDER, "images_npz")
		results = analyze_registration_quality(image, ID_image, registered_folder)

		# Encontra a melhor imagem
		best_image, best_score = find_best_registration(results)

		# Imprime os resultados
		#print("\nResultados de registro para todas as imagens:")
		#for image_name, metrics in results.items():
		#	print(f"\nImage: {image_name}")
		#	print(f"MSE: {metrics['MSE']:.4f}")
		#	print(f"NCC: {metrics['NCC']:.4f}")
		#	print(f"MI: {metrics['MI']:.4f}")

		#print(f"\nBest register: {best_image}")
		#print(f"Combined score: {best_score:.4f}")
		print('Registration completed successfully!')

		ID_template = os.path.basename(best_image).replace('.npz','')

		template_path = os.path.join(registered_folder, ID_template+'.npz')
		template_array = np.load(template_path)["image"][:].astype(np.float32)
		template_array = template_array.transpose(2,1,0)
		group = np.load(template_path)["group"]

		image_path = os.path.join(TEMP_IMAGES, 'registered_images/groups/group_'+str(group), 'npz_rigid', ID_image+'.npz')
		image_array = np.load(image_path)["image"][:].astype(np.float32)
		image_array = image_array.transpose(2,1,0)




		pre_trained_model_lung_path = 'weights/LightningLung_epoch=90-val_loss=0.014_attUnet_template_lr=0.0001_AdamW_focal_loss_kaggle_saida=6.ckpt'

		test_model_lung = LungModule.load_from_checkpoint(pre_trained_model_lung_path, strict=False)

		lung = test_model_lung.predict_lung(image_path)
		#lung = test_model_lung.predict(sample, ID_image)

		rigid_path = busca_path(ID_image, group)
		salvaImageRebuilt(lung.squeeze(), image_original_path, rigid_path=rigid_path, ID_image=ID_image, msg='lung', output_path=output_path)

if __name__ == '__main__':
	os.system('cls' if os.name == 'nt' else 'clear')
	sys.exit(main(sys.argv))
