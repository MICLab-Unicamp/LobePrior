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