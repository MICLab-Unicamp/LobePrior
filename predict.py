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
from tqdm import tqdm

from utils.general import register_single, teste_pickle_by_image, process_images
from utils.general import unified_img_reading, convert_to_nifti, remove_directories_if_exist, collect_images_verbose
from utils.general import analyze_registration_quality, find_best_registration
from utils.transform3D import CTHUClip
from predict_decoders import LoberModule
from predict_normal import LoberModuleNormal

HOME = os.getenv("HOME")
TEMP_IMAGES = 'temp_images'
RAW_DATA_FOLDER = 'raw_images' #os.path.join(HOME, 'raw_images')

def main(args):
	print('Parameters:', args)

	delete_data = False
	output_path = os.path.join(TEMP_IMAGES, 'outputs')

	parser = argparse.ArgumentParser(description='Lung lobe segmentation on CT images using prior information.')
	parser.add_argument('--input', "-i", default="inputs", help= "Input image or folder with volumetric images.", type=str)
	parser.add_argument('--output', "-o", default="outputs", help= "Directory to store the final segmentation.", type=str)
	parser.add_argument('--nworkers', "-nw", default=mp.cpu_count()//2, help="Number of workers", type=int)
	parser.add_argument('--normal', "-n", action="store_true", help= "Use Prior Information.") 			# true se passou --normal
	parser.add_argument('--delete', "-d", action="store_true", help= "Delete temporary files.") 		# true se passou --delete
	parser.add_argument('--pool', "-p", action="store_true", help= "Parallel processing.") 		# true se passou --pool

	args = parser.parse_args()

	image_original_path = args.input
	output_path = args.output
	modo_normal = args.normal
	delete_data = args.delete
	parallel_processing = args.pool
	N_THREADS = args.nworkers

	print(f'Input: {image_original_path}')
	print(f'Output: {output_path}')
	print(f'Prior Information: {not modo_normal}')
	print(f'Delete temporary files : {delete_data}')
	print(f'Parallel processing: {parallel_processing}')

	if parallel_processing:
		print(f'Number of processes: {N_THREADS}')

	'''
	if os.path.isfile(image_original_path):
		path = Path(image_original_path)
		ext = "".join(path.suffixes)
		if ext not in ['.nii', '.nii.gz', '.mhd', '.mha']:
			print(f'The file format is not valid: {ext}')
			print(f'The image name must not contain dots or the image extension must be .nii, .nii.gz, .mhd or .mha')
			#return 0
		all_images = [image_original_path]
	elif os.path.isdir(image_original_path):
		extensoes = ['*.nii', '*.nii.gz', '*.mhd', '*.mha']
		all_images = []
		for ext in extensoes:
			all_images.extend(glob.glob(os.path.join(image_original_path, ext)))
	else:
		all_images = sorted(glob.glob(os.path.join(image_original_path, '*.nii.gz')))

	print(f'Number of images found in the dataset: {len(all_images)}')
	'''

	all_images = collect_images_verbose(image_original_path)

	if len(all_images)==0:
		print('Either the image path is incorrect or the input image is missing.')
		print('python predict.py -i <input.nii.gz>')
		return 0

	for image_original_path in all_images:
		path = Path(image_original_path)
		ext = "".join(path.suffixes)
		if ext in ['.mhd', '.mha']:
			image_original_path = convert_to_nifti(image_original_path)
		ID_image = os.path.basename(image_original_path).replace('.nii.gz','').replace('.nii','').replace('.mhd','').replace('.mha','')
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
			print('Running without prior information.')
			image_path = os.path.join(TEMP_IMAGES, 'output_convert_cliped_isometric/images', ID_image+'.nii.gz')
			image_data = nib.load(image_path).get_fdata()

			npz_path = os.path.join(TEMP_IMAGES, 'npz_without_registration', f'{ID_image}.npz')
			os.makedirs(os.path.join(TEMP_IMAGES, 'npz_without_registration'), exist_ok=True)

			np.savez_compressed(npz_path, image=image_data, ID=ID_image)



			pre_trained_model_path = 'weights/LightningLobes_no_template.ckpt'

			test_model = LoberModuleNormal.load_from_checkpoint(pre_trained_model_path, strict=False)

			test_model.predict(npz_path, image_original_path, output_path, post_processed=True, save_image=True, rebuild=True)

		else:
			print('Running with prior information.')

			image_path = os.path.join(TEMP_IMAGES, 'output_convert_cliped_isometric/images', ID_image+'.nii.gz')

			process_images(image_path, ID_image, N_THREADS, parallel_processing=parallel_processing)

			'''
			if parallel_processing:
				#N_THREADS = mp.cpu_count()//2
				arg_list = []

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
			'''



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

			pre_trained_model_path = 'weights/LightningLobes.ckpt'

			test_model = LoberModule.load_from_checkpoint(pre_trained_model_path, strict=False)

			test_model.predict(image_path, image_original_path, output_path, group=group, post_processed=True)

	if delete_data:
		dirs = [
			os.path.join(TEMP_IMAGES, 'output_convert_cliped_isometric'),
			os.path.join(TEMP_IMAGES, 'npz_without_registration')
			#os.path.join(TEMP_IMAGES, 'registered_images')
		]
		remove_directories_if_exist(dirs)

	return 0

if __name__ == '__main__':
	os.system('cls' if os.name == 'nt' else 'clear')
	sys.exit(main(sys.argv))
