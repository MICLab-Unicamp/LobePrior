#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import glob
import skimage
import torch
import numpy as np
import nibabel as nib
import torchio as tio
import SimpleITK as sitk
import pytorch_lightning as pl
from matplotlib import pyplot as plt
from monai.metrics import DiceMetric
from scipy.ndimage import zoom

from dipy.align.imaffine import (MutualInformationMetric, transform_centers_of_mass, AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D, RigidTransform3D, AffineTransform3D)

from model.unet_diedre import UNet_Diedre
from utils.unified_img_reading_resample import unified_img_reading_isometric
from utils.post_processed_3D import pos_processed
from utils.to_onehot import to_onehot
from utils.metric import Dice_chavg_per_label_metric

DATA_FOLDER = os.getenv("HOME")
RAW_DATA_FOLDER = os.path.join(DATA_FOLDER, 'DataSets/preprocessed/isometric/output_convert_cliped')

RAW_DATA_FOLDER_TEMPLATE = os.path.join(DATA_FOLDER, '/home/jean/DataSets/outputs_registered_high/isometric_cliped_and_normalized')

TEMPLATE_1_PATH  = os.path.join(RAW_DATA_FOLDER_TEMPLATE, 'model_fusion/group_1.npz')
TEMPLATE_2_PATH  = os.path.join(RAW_DATA_FOLDER_TEMPLATE, 'model_fusion/group_2.npz')
TEMPLATE_3_PATH  = os.path.join(RAW_DATA_FOLDER_TEMPLATE, 'model_fusion/group_3.npz')
TEMPLATE_4_PATH  = os.path.join(RAW_DATA_FOLDER_TEMPLATE, 'model_fusion/group_4.npz')
TEMPLATE_5_PATH  = os.path.join(RAW_DATA_FOLDER_TEMPLATE, 'model_fusion/group_5.npz')
TEMPLATE_6_PATH  = os.path.join(RAW_DATA_FOLDER_TEMPLATE, 'model_fusion/group_6.npz')
TEMPLATE_7_PATH  = os.path.join(RAW_DATA_FOLDER_TEMPLATE, 'model_fusion/group_7.npz')
TEMPLATE_8_PATH  = os.path.join(RAW_DATA_FOLDER_TEMPLATE, 'model_fusion/group_8.npz')
TEMPLATE_9_PATH  = os.path.join(RAW_DATA_FOLDER_TEMPLATE, 'model_fusion/group_9.npz')
TEMPLATE_10_PATH = os.path.join(RAW_DATA_FOLDER_TEMPLATE, 'model_fusion/group_10.npz')
TEMPLATE_11_PATH = os.path.join(RAW_DATA_FOLDER_TEMPLATE, 'model_fusion/group_11.npz')

def show_output_with_background(image, epoch):
	f, (plot0, plot1, plot2, plot3, plot4, plot5) = plt.subplots(1, 6, figsize = (12, 6))

	plot0.imshow(image[0,0][image.shape[2]//2])
	plot0.set_axis_off()
	plot1.imshow(image[0,1][image.shape[2]//2])
	plot1.set_axis_off()
	plot2.imshow(image[0,2][image.shape[2]//2])
	plot2.set_axis_off()
	plot3.imshow(image[0,3][image.shape[2]//2])
	plot3.set_axis_off()
	plot4.imshow(image[0,4][image.shape[2]//2])
	plot4.set_axis_off()
	plot5.imshow(image[0,5][image.shape[2]//2])
	plot5.set_axis_off()
	#save_path = os.path.join('output_with_backgound'+str(epoch)+'.png')
	#plt.savefig(save_path, dpi=300, transparent=True)
	plt.show()
	plt.close()

def show_output(image, epoch):
	f, (plot1, plot2, plot3, plot4, plot5) = plt.subplots(1, 5, figsize = (12, 6))
	plot1.imshow(image[0,0][image.shape[2]//2])
	plot1.set_axis_off()
	plot2.imshow(image[0,1][image.shape[2]//2])
	plot2.set_axis_off()
	plot3.imshow(image[0,2][image.shape[2]//2])
	plot3.set_axis_off()
	plot4.imshow(image[0,3][image.shape[2]//2])
	plot4.set_axis_off()
	plot5.imshow(image[0,4][image.shape[2]//2])
	plot5.set_axis_off()
	#save_path = os.path.join('output_'+str(epoch)+'.png')
	#plt.savefig(save_path, dpi=300, transparent=True)
	plt.show()
	plt.close()

def show_images(image, output):
	print(image.shape, output.shape)

	f, (plot1, plot2, plot3, plot4, plot5, plot6) = plt.subplots(1, 6, figsize = (12, 6))
	plot1.imshow(image[0,0][image.shape[2]//2])
	plot1.set_axis_off()
	plot2.imshow(image[0,0][output.shape[2]//2])
	plot2.set_axis_off()
	plot3.imshow(image[0,0][:,image.shape[3]//2])
	plot3.set_axis_off()
	plot4.imshow(image[0,0][:,output.shape[3]//2])
	plot4.set_axis_off()
	plot5.imshow(image[0,0][:,:,image.shape[3]//2])
	plot5.set_axis_off()
	plot6.imshow(image[0,0][:,:,output.shape[4]//2])
	plot6.set_axis_off()
	plt.show()
	plt.close()

def register(moving_path, moving_label_path=None, moving_lung_path=None, group=1):
	GROUP = group

	OUTPUT_DIR = os.path.join(DATA_FOLDER, "isbi_jean/results/registered_images/outputs_images_and_labels/group_"+str(GROUP))
	OUTPUT_DIR_IMAGES = 'images_nao_corrigidas'
	OUTPUT_DIR_LABELS = 'labels_nao_corrigidas'
	OUTPUT_DIR_LUNGS = 'lungs_nao_corrigidas'
	OUTPUT_RIGID_IMG = os.path.join(OUTPUT_DIR, 'rigid3D', OUTPUT_DIR_IMAGES)
	OUTPUT_RIGID_TGT = os.path.join(OUTPUT_DIR, 'rigid3D', OUTPUT_DIR_LABELS)
	OUTPUT_RIGID_LUNG =  os.path.join(OUTPUT_DIR, 'rigid3D', OUTPUT_DIR_LUNGS)
	OUTPUT_AFFINE_IMG = os.path.join(OUTPUT_DIR, 'affine3D', OUTPUT_DIR_IMAGES)
	OUTPUT_AFFINE_TGT = os.path.join(OUTPUT_DIR, 'affine3D', OUTPUT_DIR_LABELS)
	OUTPUT_AFFINE_LUNG =  os.path.join(OUTPUT_DIR, 'affine3D', OUTPUT_DIR_LUNGS)
	OUTPUT_DIR_NPZ_RIGID = os.path.join(OUTPUT_DIR, 'npz_rigid')
	OUTPUT_DIR_NPZ_AFFINE = os.path.join(OUTPUT_DIR, 'npz_affine')

	os.makedirs(OUTPUT_RIGID_IMG, exist_ok=True)
	os.makedirs(OUTPUT_RIGID_TGT, exist_ok=True)
	os.makedirs(OUTPUT_RIGID_LUNG, exist_ok=True)
	#os.makedirs(OUTPUT_AFFINE_IMG, exist_ok=True)
	#os.makedirs(OUTPUT_AFFINE_TGT, exist_ok=True)
	#os.makedirs(OUTPUT_AFFINE_LUNG, exist_ok=True)
	os.makedirs(OUTPUT_DIR_NPZ_RIGID, exist_ok=True)
	##os.makedirs(OUTPUT_DIR_NPZ_AFFINE, exist_ok=True)

	if GROUP==1:
		fixed_path = os.path.join(RAW_DATA_FOLDER, 'groups/group_'+str(GROUP)+'/1.3.6.1.4.1.14519.5.2.1.6279.6001.192256506776434538421891524301.nii.gz')
	elif GROUP==2:
		fixed_path = os.path.join(RAW_DATA_FOLDER, 'groups/group_'+str(GROUP)+'/1.3.6.1.4.1.14519.5.2.1.6279.6001.194465340552956447447896167830.nii.gz')
	elif GROUP==3:
		fixed_path = os.path.join(RAW_DATA_FOLDER, 'groups/group_'+str(GROUP)+'/1.3.6.1.4.1.14519.5.2.1.6279.6001.183843376225716802567192412456.nii.gz')
	elif GROUP==4:
		fixed_path = os.path.join(RAW_DATA_FOLDER, 'groups/group_'+str(GROUP)+'/1.3.6.1.4.1.14519.5.2.1.6279.6001.265133389948279331857097127422.nii.gz')
	elif GROUP==5:
		fixed_path = os.path.join(RAW_DATA_FOLDER, 'groups/group_'+str(GROUP)+'/1.3.6.1.4.1.14519.5.2.1.6279.6001.316911475886263032009840828684.nii.gz')
	elif GROUP==6:
		fixed_path = os.path.join(RAW_DATA_FOLDER, 'groups/group_'+str(GROUP)+'/1.3.6.1.4.1.14519.5.2.1.6279.6001.842980983137518332429408284002.nii.gz')
	elif GROUP==7:
		fixed_path = os.path.join(RAW_DATA_FOLDER, 'groups/group_'+str(GROUP)+'/1.3.6.1.4.1.14519.5.2.1.6279.6001.177685820605315926524514718990.nii.gz')
	elif GROUP==8:
		fixed_path = os.path.join(RAW_DATA_FOLDER, 'groups/group_'+str(GROUP)+'/coronacases_005.nii.gz')
	elif GROUP==9:
		fixed_path = os.path.join(RAW_DATA_FOLDER, 'groups/group_'+str(GROUP)+'/1.3.6.1.4.1.14519.5.2.1.6279.6001.188385286346390202873004762827.nii.gz')
	elif GROUP==10:
		fixed_path = os.path.join(RAW_DATA_FOLDER, 'groups/group_'+str(GROUP)+'/1.3.6.1.4.1.14519.5.2.1.6279.6001.249530219848512542668813996730.nii.gz')
	elif GROUP==11:
		fixed_path = os.path.join(RAW_DATA_FOLDER, 'groups/group_'+str(GROUP)+'/1.3.6.1.4.1.14519.5.2.1.6279.6001.300392272203629213913702120739.nii.gz')

	print('Fixed path', fixed_path)

	ID_fixed = os.path.basename(fixed_path).replace(".nii.gz", '')
	ID_moving = os.path.basename(moving_path).replace(".nii.gz", '')
	print(f'\n{ID_moving}')

	fixed_label_path = os.path.join(RAW_DATA_FOLDER, 'labels', ID_fixed+'.nii.gz')

	template_img = nib.load(fixed_path)
	template_label_img = nib.load(fixed_label_path)

	template_data = template_img.get_fdata()
	template_grid2world = template_img.affine
	template_label_data = template_label_img.get_fdata()

	if ID_moving != ID_fixed:
		moving_img = nib.load(moving_path)
		moving_label = nib.load(moving_label_path)
		moving_lung = nib.load(moving_lung_path)

		moving_data = moving_img.get_fdata()
		moving_grid2world = moving_img.affine
		moving_label_data = moving_label.get_fdata()
		moving_lung_data = moving_lung.get_fdata()
		print(moving_data.shape, moving_lung_data.shape, template_data.shape)




		c_of_mass = transform_centers_of_mass(template_data, template_grid2world, moving_data, moving_grid2world)

		print('Transform centers of mass realizado!')

		nbins = 32
		sampling_prop = None
		metric = MutualInformationMetric(nbins, sampling_prop)

		level_iters = [10000, 1000, 100]
		sigmas = [3.0, 1.0, 0.0]
		factors = [4, 2, 1]

		affreg = AffineRegistration(metric=metric, level_iters=level_iters, sigmas=sigmas, factors=factors)

		transform = TranslationTransform3D()
		params0 = None
		starting_affine = c_of_mass.affine
		translation = affreg.optimize(template_data, moving_data, transform, params0, template_grid2world, moving_grid2world, starting_affine=starting_affine)

		print('TranslationTransform3D realizado!')

		transform = RigidTransform3D()
		params0 = None
		starting_affine = translation.affine
		rigid = affreg.optimize(template_data, moving_data, transform, params0, template_grid2world, moving_grid2world, starting_affine=starting_affine)

		transformed = rigid.transform(moving_data)
		if moving_label_path:
			transformed_label = rigid.transform(moving_label_data, interpolation='nearest')
		if moving_lung:
			transformed_lung = rigid.transform(moving_lung_data, interpolation='nearest')

		img = nib.Nifti1Image(transformed, rigid.affine)
		nib.save(img, os.path.join(OUTPUT_RIGID_IMG, ID_moving+'_rigid3D.nii.gz'))
		tgt = nib.Nifti1Image(transformed_label, rigid.affine)
		nib.save(tgt, os.path.join(OUTPUT_RIGID_TGT, ID_moving+'_rigid3D.nii.gz'))
		lung = nib.Nifti1Image(transformed_lung, rigid.affine)
		nib.save(lung, os.path.join(OUTPUT_RIGID_LUNG, ID_moving+'_rigid3D.nii.gz'))

		save_path = os.path.join(OUTPUT_DIR_NPZ_RIGID, f'{ID_moving}_rigid3D.npz')
		np.savez_compressed(save_path, image=transformed, label=transformed_label, lung=transformed_lung, ID=ID_moving, ct_path=moving_path, mask_path=moving_label_path, target_name="lobes")

		print('RigidTransform3D realizado!')

		'''
		transform = AffineTransform3D()
		params0 = None
		starting_affine = rigid.affine
		affine = affreg.optimize(template_data, moving_data, transform, params0, template_grid2world, moving_grid2world, starting_affine=starting_affine)

		transformed = affine.transform(moving_data)
		if moving_label:
			transformed_label = affine.transform(moving_label_data, interpolation='nearest')
		if moving_lung:
			transformed_lung = rigid.transform(moving_lung_data, interpolation='nearest')

		img = nib.Nifti1Image(transformed, affine.affine)
		nib.save(img, os.path.join(OUTPUT_AFFINE_IMG, ID_moving+'_affine3D.nii.gz'))
		tgt = nib.Nifti1Image(transformed_label, affine.affine)
		nib.save(tgt, os.path.join(OUTPUT_AFFINE_TGT, ID_moving+'_affine3D.nii.gz'))
		lung = nib.Nifti1Image(transformed_lung, affine.affine)
		nib.save(tgt, os.path.join(OUTPUT_AFFINE_LUNG, ID_moving+'_affine3D.nii.gz'))

		save_path = os.path.join(OUTPUT_DIR_NPZ_AFFINE, f'{ID_moving}_affine3D.npz')
		np.savez_compressed(save_path, image=transformed, label=transformed_label, lung=transformed_lung, ID=ID_moving, ct_path=moving_path, mask_path=moving_label_path, target_name="lobes")

		print('AffineTransform3D realizado!')
		'''
	else:
		print('moving e fixed iguais')
		'''
		img = nib.Nifti1Image(template_data, template_img.affine)
		nib.save(img, os.path.join(OUTPUT_RIGID_IMG, ID_fixed+'_rigid3D.nii.gz'))
		tgt = nib.Nifti1Image(template_label_data, template_label_img.affine)
		nib.save(tgt, os.path.join(OUTPUT_RIGID_TGT, ID_fixed+'_rigid3D.nii.gz'))

		img = nib.Nifti1Image(template_data, template_img.affine)
		nib.save(img, os.path.join(OUTPUT_AFFINE_IMG, ID_fixed+'_affine3D.nii.gz'))
		tgt = nib.Nifti1Image(template_label_data, template_label_img.affine)
		nib.save(tgt, os.path.join(OUTPUT_AFFINE_TGT, ID_fixed+'_affine3D.nii.gz'))

		save_path = os.path.join(OUTPUT_DIR_NPZ_RIGID, f'{ID_moving}_rigid3D.npz')
		np.savez_compressed(save_path, image=template_data, label=template_label_data, ID=ID_fixed, ct_path=fixed_path, mask_path=fixed_label_path, target_name="lobes")
		save_path = os.path.join(OUTPUT_DIR_NPZ_AFFINE, f'{ID_moving}_affine3D.npz')
		np.savez_compressed(save_path, image=template_data, label=template_label_data, ID=ID_fixed, ct_path=fixed_path, mask_path=fixed_label_path, target_name="lobes")
		'''

	return transformed

def main(args):
	all_images = sorted(glob.glob(os.path.join(DATA_FOLDER, 'isbi_jean/results/converted_images/images/coronacases', '*.nii.gz')))
	print('Quantidades de imagens encontradas no dataset:', len(all_images))

	for group in range(1,12):
		for image_path in all_images:
			ID = os.path.basename(image_path).replace('.nii.gz','').replace('.nii','').replace('.npz','').replace('_affine3D','').replace('_rigid3D','')

			lung_path = image_path.replace('images/','lungs/')
			label_path = os.path.join(DATA_FOLDER, 'dados/segmentations/lungmask/LTRCLobes_R231',ID+'.nii.gz')

			register(image_path, moving_label_path=label_path, moving_lung_path=lung_path, group=group)

	return 0

if __name__ == '__main__':
	os.system('cls' if os.name == 'nt' else 'clear')
	sys.exit(main(sys.argv))
