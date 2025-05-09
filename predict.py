#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import argparse
import torchvision
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import multiprocessing as mp
from pathlib import Path

from utils.general import register_single, teste_pickle_by_image
from utils.general import unified_img_reading
from utils.general import analyze_registration_quality, find_best_registration
from utils.transform3D import CTHUClip
from predict_decoders import LoberModule
from predict_normal import LoberModuleNormal

HOME = os.getenv("HOME")
TEMP_IMAGES = 'temp_images'
RAW_DATA_FOLDER = 'raw_images' #os.path.join(HOME, 'raw_images')

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
	parser.add_argument('--pool', "-p", action="store_true", help= "Parallel processing.") 		# true se passou --pool

	args = parser.parse_args()

	image_original_path = args.input
	output_path = args.output
	modo_normal = args.normal
	delete_data = args.delete
	parallel_processing = args.pool

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

	if len(all_images)==0:
		print('Either the image path is incorrect or the input image is missing.')
		print('python predict.py -i <input.nii.gz>')

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




		if modo_normal:
			image_path = os.path.join(TEMP_IMAGES, 'output_convert_cliped_isometric/images', ID_image+'.nii.gz')
			image_data = nib.load(image_path).get_fdata()

			npz_path = os.path.join(TEMP_IMAGES, 'npz_without_registration', f'{ID_image}.npz')
			os.makedirs(os.path.join(TEMP_IMAGES, 'npz_without_registration'), exist_ok=True)

			np.savez_compressed(npz_path, image=image_data, ID=ID_image)



			pre_trained_model_path = 'weights/LightningLobes_6_decoders_no_attUnet_template_epoch=39-val_loss=0.249_attUnet_template_lr=0.0001_AdamW_focal_loss_kaggle_saida=6.ckpt'

			test_model = LoberModuleNormal.load_from_checkpoint(pre_trained_model_path, strict=False)

			test_model.predict(npz_path, image_original_path, output_path, post_processed=True, save_image=True, rebuild=True)

			if delete_data:
				os.rmdir(os.path.join(TEMP_IMAGES, 'output_convert_cliped_isometric'))
				os.rmdir(os.path.join(TEMP_IMAGES, 'npz_without_registration'))
		else:
			image_path = os.path.join(TEMP_IMAGES, 'output_convert_cliped_isometric/images', ID_image+'.nii.gz')

			if parallel_processing:
				N_THREADS = mp.cpu_count()//2
				arg_list = []

				image_path = os.path.join(TEMP_IMAGES, 'output_convert_cliped_isometric/images', ID_image+'.nii.gz')

				for group in range(1,11):
					if teste_pickle_by_image(ID_image, group)==False:
						arg_list.append((image_path, None, None, None, group))

				with mp.Pool(N_THREADS) as pool:
					results = list(tqdm(pool.starmap(register_single, arg_list), total=len(arg_list)))
			else:
				for group in range(1,11):
					if teste_pickle_by_image(ID_image, group)==False:
						register_single(image_path, None, None, None, group)

			print('Registration completed successfully!')



			image = nib.load(image_path).get_fdata()

			# Analisa todas as imagens
			registered_folder = os.path.join(RAW_DATA_FOLDER, "images_npz")
			results = analyze_registration_quality(image, ID_image, registered_folder)

			# Encontra a melhor imagem
			best_image, best_score = find_best_registration(results)

			# Imprime os resultados
			#print("\nRegistration results for all images:")
			#for image_name, metrics in results.items():
			#	print(f"\nImage: {image_name}")
			#	print(f"MSE: {metrics['MSE']:.4f}")
			#	print(f"NCC: {metrics['NCC']:.4f}")
			#	print(f"MI: {metrics['MI']:.4f}")

			#print(f"\nBest register: {best_image}")
			#print(f"Combined score: {best_score:.4f}")
			print("Best registration successfully found!")

			ID_template = os.path.basename(best_image).replace('.npz','')

			template_path = os.path.join(registered_folder, ID_template+'.npz')
			template_array = np.load(template_path)["image"][:].astype(np.float32)
			template_array = template_array.transpose(2,1,0)
			group = np.load(template_path)["group"]

			image_path = os.path.join(TEMP_IMAGES, 'registered_images/groups/group_'+str(group), 'npz_rigid', ID_image+'.npz')
			image_array = np.load(image_path)["image"][:].astype(np.float32)
			image_array = image_array.transpose(2,1,0)

			pre_trained_model_path = 'weights/LightningLobes_6_decoders_pre_treino_epoch=62-val_loss=0.145_attUnet_template_lr=0.0001_AdamW_focal_loss_kaggle_saida=6.ckpt'

			test_model = LoberModule.load_from_checkpoint(pre_trained_model_path, strict=False)

			test_model.predict(image_path, image_original_path, output_path, group=group, post_processed=True)

			if delete_data:
				os.rmdir(os.path.join(TEMP_IMAGES, 'output_convert_cliped_isometric'))
				os.rmdir(os.path.join(TEMP_IMAGES, 'registered_images'))

	return 0

if __name__ == '__main__':
	os.system('cls' if os.name == 'nt' else 'clear')
	sys.exit(main(sys.argv))
