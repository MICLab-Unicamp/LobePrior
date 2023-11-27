#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import SimpleITK as sitk
import pydicom as pyd
from tqdm import tqdm

def show_metrics(dice, dice_per_label):
	#print(f"Dice: {dice}")
	#print('Dice: {:.3f}'.format(dice))
	#print(f"Jaccard do batch: {jaccard}")
	#print('DSC:',np.mean(dice_per_label), [float(x) for x in dice_per_label])
	#print('Jaccard:',np.mean(jaccard_per_label), [float(x) for x in jaccard_per_label])
	#print('DSC:',dsc, np.mean(dice_per_label), dice_per_label)
	#print('Jaccard:',jaccard, np.mean(jaccard_per_label), jaccard_per_label)
	#print(f'\tDice: {np.mean(dice_per_label):.3f} {np.mean(dice_per_label[1:]):.3f}',' '.join(['%.3f' % (x) for x in dice_per_label]))
	#print(f'\tDice: {np.mean(dice_per_label):.3f}, {np.mean(dice_per_label[1:]):.3f},',', '.join(['%.3f' % (x) for x in dice_per_label]))
	print(f'{np.mean(dice_per_label):.3f}, {np.mean(dice_per_label[1:]):.3f},',', '.join(['%.3f' % (x) for x in dice_per_label]))
	#print(f'\tDice: {np.mean(dice_per_label):.3f} & {np.mean(dice_per_label[1:]):.3f} &',' & '.join(['%.3f' % (x) for x in dice_per_label]))
	#print(f'\tDice sem background: {np.mean(dice_per_label[1:]):.3f}')

class SalvaFile():
	def __init__(self, resutls_path='../../logs/save'):
		self.methods = ['dice','false_negative_error','false_positive_error','jaccard','volume_similarity','abs_volume_similarity','avg_hd','hd','avg_hd','hd']
		self.resutls_path = resutls_path
		os.makedirs(self.resutls_path, exist_ok=True)
		for method in self.methods:
			f = open(self.resutls_path+'/'+method+'.csv','w')
			for name in ['STD','MEAN','BG','LUL','LLL','RUL','RML','RLL']:
				f.write(name+';')
			f.write('\n')
			f.close()

	def salva_arq(self, metrics, struct_names=['Lung']):
		for method in self.methods:
			f = open(self.resutls_path+'/'+method+'.csv','a')
			list = []
			for name in ['LUL','LLL','RUL','RML','RLL']:
				list.append(metrics[name][method])

			std = str(np.std(list)).replace(',',';').replace('.',',')
			mean = str(np.mean(list)).replace(',',';').replace('.',',')
			f.write(std +';'+mean+';')
			for name in struct_names:
				value = str(metrics[name][method]).replace('.',',').replace('[','').replace(']','')
				f.write(str(value)+';')
			f.write('\n')
			f.close()

def read_dicoms(path, primary=True, original=True):
	allfnames = []
	for dir, _, fnames in os.walk(path):
		[allfnames.append(os.path.join(dir, fname)) for fname in fnames]

	dcm_header_info = []
	dcm_parameters = []
	unique_set = []  # need this because too often there are duplicates of dicom files with different names
	i = 0
	for fname in tqdm(allfnames):
		filename_ = os.path.splitext(os.path.split(fname)[1])
		i += 1
		if filename_[0] != 'DICOMDIR':
			try:
				dicom_header = pyd.dcmread(fname, defer_size=100, stop_before_pixels=True, force=True)
				if dicom_header is not None:
					if 'ImageType' in dicom_header:
						if primary:
							is_primary = all([x in dicom_header.ImageType for x in ['PRIMARY']])
						else:
							is_primary = True

						if original:
							is_original = all([x in dicom_header.ImageType for x in ['ORIGINAL']])
						else:
							is_original = True

						# if 'ConvolutionKernel' in dicom_header:
						#     ck = dicom_header.ConvolutionKernel
						# else:
						#     ck = 'unknown'
						if is_primary and is_original and 'LOCALIZER' not in dicom_header.ImageType:
							h_info_wo_name = [dicom_header.StudyInstanceUID, dicom_header.SeriesInstanceUID,
											  dicom_header.ImagePositionPatient]
							h_info = [dicom_header.StudyInstanceUID, dicom_header.SeriesInstanceUID, fname,
									  dicom_header.ImagePositionPatient]
							if h_info_wo_name not in unique_set:
								unique_set.append(h_info_wo_name)
								dcm_header_info.append(h_info)
								# kvp = None
								# if 'KVP' in dicom_header:
								#     kvp = dicom_header.KVP
								# dcm_parameters.append([ck, kvp,dicom_header.SliceThickness])
			except:
				print("Unexpected error:", sys.exc_info()[0])
				print("Doesn't seem to be DICOM, will be skipped: ", fname)

	conc = [x[1] for x in dcm_header_info]
	sidx = np.argsort(conc)
	conc = np.asarray(conc)[sidx]
	dcm_header_info = np.asarray(dcm_header_info, dtype=object)[sidx]
	# dcm_parameters = np.asarray(dcm_parameters)[sidx]
	vol_unique = np.unique(conc, return_index=1, return_inverse=1)  # unique volumes
	n_vol = len(vol_unique[1])
	if n_vol == 1:
		print('There is ' + str(n_vol) + ' volume in the study')
	else:
		print('There are ' + str(n_vol) + ' volumes in the study')

	relevant_series = []
	relevant_volumes = []

	for i in range(len(vol_unique[1])):
		curr_vol = i
		info_idxs = np.where(vol_unique[2] == curr_vol)[0]
		vol_files = dcm_header_info[info_idxs, 2]
		positions = np.asarray([np.asarray(x[2]) for x in dcm_header_info[info_idxs, 3]])
		slicesort_idx = np.argsort(positions)
		vol_files = vol_files[slicesort_idx]
		relevant_series.append(vol_files)
		reader = sitk.ImageSeriesReader()
		reader.SetFileNames(vol_files)
		vol = reader.Execute()
		relevant_volumes.append(vol)

	return relevant_volumes
