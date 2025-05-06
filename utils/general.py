#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cc3d
import pickle
import pydicom
import torch
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from operator import itemgetter
from pathlib import Path
from collections import Counter
from typing import List, Tuple
from scipy.ndimage import zoom
from scipy.ndimage import distance_transform_edt
from dipy.align.imaffine import (MutualInformationMetric, transform_centers_of_mass, AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D, RigidTransform3D)

TEMP_IMAGES = 'temp_images'
RAW_DATA_FOLDER = 'raw_images' #os.path.join(HOME, 'raw_images')

def get_connected_components(volume, return_largest=2, verbose=False):
	'''
	volume: input volume
	return_largest: how many of the largest labels to return. If 0, nothing is changed in input volume
	verbose: prints label_count
	returns:
		filtered_volume, label_count, labeled_volume
	'''
	labels_out = cc3d.connected_components(volume.astype(np.int32))
	#print(labels_out)
	label_count = np.unique(labels_out, return_counts=True)[1]

	# Indicate which was the original label and sort by count
	label_count = [(label, count) for label, count in enumerate(label_count)]
	label_count.sort(key=itemgetter(1), reverse=True)
	label_count.pop(0)  # remove largest which should be background

	if verbose:
		print(f"Label count: {label_count}")

	filtered = None
	if return_largest > 0:
		for i in range(return_largest):
			try:
				id_max = label_count[i][0]
				if filtered is None:
					filtered = (labels_out == id_max)
				else:
					filtered += (labels_out == id_max)
			except IndexError:
				# We want more components that what is in the image, stop
				break

		#print(filtered)
		#print(volume)
		volume = filtered * volume
		labels_out = filtered * labels_out

	return volume, label_count, labels_out

def post_processing_lung(output, largest=1, verbose=None):
	'''
	Post processing pipeline for lung segmentation only
	Input should be numpy activations
	'''
	assert output.ndim == 3, "Input to lung post processing has to be three dimensional no channels"

	if verbose is not None:
		verbose.write("Unpacking outputs...")
	lung = (output > 0.5).astype(np.int32)
	if verbose is not None:
		verbose.write("Calculating lung connected components...")
	lung, lung_lc, lung_labeled = get_connected_components(lung, return_largest=largest)

	if verbose is not None:
		verbose.write("Extracting first and second largest components...")
	first_component = lung_labeled == lung_lc[0][0]
	try:
		second_component = lung_labeled == lung_lc[1][0]
	except IndexError:
		print("WARNING: Was not able to get a second component.")
		second_component = np.zeros_like(first_component)

	if verbose is not None:
		verbose.write("WARNING: Skipping lung split.")
	lung = first_component + second_component

	return lung.astype(np.uint8)

def assign_to_closest_label(image: np.ndarray,
						  target_label: int = 6,
						  valid_labels: List[int] = [1, 2, 3, 4, 5]) -> Tuple[np.ndarray, dict]:
	"""
	Atribui voxels com uma determinada label à label mais próxima dentre um conjunto de labels válidas.

	Args:
		image: Array 3D numpy com as labels
		target_label: Label que será reatribuída (default: 6)
		valid_labels: Lista de labels válidas para atribuição (default: [1,2,3,4,5])

	Returns:
		Tuple contendo:
		- Imagem com as novas labels atribuídas
		- Dicionário com estatísticas sobre as mudanças
	"""
	# Cria uma cópia da imagem para não modificar a original
	result = image.copy()

	# Cria uma máscara para os voxels que precisam ser reatribuídos
	target_mask = (image == target_label)

	if not np.any(target_mask):
		return result, {"changed_voxels": 0}

	# Inicializa array para armazenar as distâncias mínimas
	min_distances = np.inf * np.ones_like(image, dtype=float)
	closest_labels = np.zeros_like(image)

	# Para cada label válida
	for label in valid_labels:
		# Cria máscara para a label atual
		label_mask = (image == label)

		if not np.any(label_mask):
			continue

		# Calcula mapa de distância para a label atual
		distances = distance_transform_edt(~label_mask)

		# Atualiza as distâncias mínimas e labels mais próximas
		update_mask = distances < min_distances
		min_distances[update_mask] = distances[update_mask]
		closest_labels[update_mask] = label

	# Atribui os voxels à label mais próxima
	result[target_mask] = closest_labels[target_mask]

	# Coleta estatísticas
	stats = {
		"changed_voxels": np.sum(target_mask),
		"distribution": {
			label: np.sum((result == label) & target_mask)
			for label in valid_labels
		}
	}

	return result, stats

def post_processing_dist_lung(image, lung, max_value=None):
	segmentation_filled = image.copy()

	if lung.max()>1:
		print(f'Lung com label maior que 1 {lung.shape}: {lung.min()} e {lung.max()}')
		lung[lung>1]=1

	if max_value is None:
		max_value = segmentation_filled.max()+1

	if torch.is_tensor(lung):
		lung = lung.numpy().astype(np.uint8)

	# preenche os lobos com valores iguais a 6, onde no pulmão é 1 e no lobo é 0
	segmentation_filled[(segmentation_filled == 0) & (lung == 1)] = 6

	# Preencher buracos na segmentação
	segmentation_filled = assign_to_closest_label(segmentation_filled, target_label=max_value)[0]

	return segmentation_filled

def pos_processamento(output, template, segmentation=None):
	#print(output.shape, template.shape, segmentation.shape)

	output = output * segmentation  # B, C, Z, Y, X * B, 1, Z, Y, Xd

	template = template.argmax(dim=1)
	output = output.argmax(dim=1)
	segmentation = segmentation[:, 0]
	output = torch.where((segmentation == 1)&(output == 0), template, output)

	return output.cpu()

def get_orientation(image_path):
	# Carrega a imagem
	img = nib.load(image_path)

	# Obtém a matriz de rotação da imagem
	aff_matrix = img.affine
	# Calcula os vetores das direções das coordenadas
	directions = np.array(aff_matrix[:3, :3])

	# Verifica o sinal do determinante da matriz de direções
	# RAS: Determinante positivo
	# PIL: Determinante negativo
	determinant = np.linalg.det(directions)

	# Define as orientações
	orientations = {
		"RAS": determinant > 0,
		"LAI": all(determinant * np.array([1, 1, -1]) > 0),
		"RAI": all(determinant * np.array([-1, 1, -1]) > 0),
		"LAS": all(determinant * np.array([1, -1, -1]) > 0),
		"LPI": all(determinant * np.array([1, 1, 1]) > 0),
		"RPI": all(determinant * np.array([-1, 1, 1]) > 0),
		"LPS": all(determinant * np.array([1, -1, 1]) > 0),
		"RPS": all(determinant * np.array([-1, -1, 1]) > 0),
		"PIL": determinant < 0
	}

	# Retorna as orientações encontradas
	return [orientation for orientation, value in orientations.items() if value]

def rebuild_output(output_path, original_path, rigid_path, ID_image, path_save=None):
	output = nib.load(output_path).get_fdata()

	if rigid_path is None:
		#print('rigid_path não existe!', rigid_path)

		original = nib.load(original_path).get_fdata()
		original = torch.from_numpy(original).float()

		output = torch.from_numpy(output).unsqueeze(dim=0).unsqueeze(dim=0)

		#print('Shape:', original.shape, output.shape)

		image_resize = torch.nn.functional.interpolate(output, size=original.shape)
		image_resize = image_resize.squeeze().numpy()
	elif os.path.exists(rigid_path):

		with open(rigid_path, 'rb') as matrix_file:
			rigid = pickle.load(matrix_file)

		output_inv = rigid.transform_inverse(output, interpolation='nearest')
		output_inv = torch.from_numpy(output_inv).unsqueeze(dim=0).unsqueeze(dim=0)
		#print('Output shape:', output_inv.shape)

		original = nib.load(original_path).get_fdata()
		original = torch.from_numpy(original).float()

		#print('Shape:', original.shape, output_inv.shape)

		image_resize = torch.nn.functional.interpolate(output_inv, size=original.shape)
		image_resize = image_resize.squeeze().numpy()

		#if dataset_name=='coronacases' or dataset_name=='lung':
		#	image_resize = np.rot90(image_resize,1).transpose(1,0,2)

		original = original.numpy()

	orientations = get_orientation(original_path)
	if 'PIL' in orientations:
		image_resize = np.rot90(image_resize,1).transpose(1,0,2)

	assert image_resize.shape==original.shape, f'The images must be the same: {image_resize.shape}=={original.shape}'

	output = nib.Nifti1Image(image_resize, nib.load(original_path).affine)

	if path_save is None:
		path_save = os.path.join(TEMP_IMAGES, 'results/images_reconstruidas')

		os.makedirs(path_save, exist_ok=True)
		print(f'Directory {path_save} created successfully.')

		nib.save(output, os.path.join(path_save, ID_image+'.nii.gz'))
	else:
		nib.save(output, path_save)

def salvaImageRebuilt(output, image_original_path, rigid_path, ID_image, msg=None, output_path=None):
	if output_path is None:
		output_path = os.path.join(TEMP_IMAGES, 'results/outputs')

	os.makedirs(output_path, exist_ok=True)
	print(f'Directory {output_path} created successfully.')

	if len(output.shape)!=3:
		if isinstance(output, np.ndarray):
			output = torch.from_numpy(output)

		if len(output.shape)==5 and output.shape[1]>1:
			output = output.squeeze().argmax(dim=0)
		elif len(output.shape)==5 and output.shape[1]==1:
			output = output.squeeze().squeeze()
		elif len(output.shape)==4:
			if output.shape[0]>1:
				print(output.shape)
				output = output.argmax(dim=0)
			else:
				output = output.squeeze()

	if len(output.shape)==3:
		if torch.is_tensor(output):
			output = output.numpy()

		output = sitk.GetImageFromArray(output)

		if msg is None:
			output_path = os.path.join(output_path, ID_image+"_lobes.nii.gz")
			sitk.WriteImage(output, output_path)
		else:
			output_path = os.path.join(output_path, ID_image+'_'+str(msg)+".nii.gz")
			sitk.WriteImage(output, output_path)
		print(f'Saving image with final post-processing in {output_path}.')
	else:
		print(f'Error: {output.shape}')

	rebuild_output(output_path, original_path=image_original_path, rigid_path=rigid_path, ID_image=ID_image, path_save=output_path)

def find_best_registration(results):
	"""
	Determina qual imagem teve o melhor registro baseado nas métricas calculadas

	Args:
		results: Dicionário com os resultados das métricas para cada imagem

	Returns:
		tuple: (melhor_imagem, pontuacao_combinada)
	"""
	scores = {}

	for image_name, metrics in results.items():
		# Menor MSE é melhor
		mse_score = 1 / (1 + metrics['MSE'])
		# Maior NCC é melhor (já está entre -1 e 1)
		ncc_score = (metrics['NCC'] + 1) / 2  # Normaliza para 0-1
		# Maior MI é melhor
		mi_score = metrics['MI']

		# Combina as métricas (pode ajustar os pesos conforme necessário)
		combined_score = (0.3 * mse_score + 0.4 * ncc_score + 0.3 * mi_score)
		scores[image_name] = combined_score

		#print('scores:', scores)

	best_image = max(scores.items(), key=lambda x: x[1])

	return best_image

def calculate_similarity_metrics(fixed_image, moving_image):
	"""
	Calcula múltiplas métricas de similaridade entre duas imagens 3D

	Args:
		fixed_image: Array numpy 3D da imagem fixa
		moving_image: Array numpy 3D da imagem em movimento (registrada)

	Returns:
		dict: Dicionário com as métricas calculadas
	"""
	# Normaliza as imagens para terem valores entre 0 e 1
	fixed_norm = (fixed_image - fixed_image.min()) / (fixed_image.max() - fixed_image.min())
	moving_norm = (moving_image - moving_image.min()) / (moving_image.max() - moving_image.min())

	# 1. Erro Médio Quadrático (MSE)
	mse = np.mean((fixed_norm - moving_norm) ** 2)

	# 2. Correlação Cruzada Normalizada (NCC)
	fixed_mean = fixed_norm.mean()
	moving_mean = moving_norm.mean()
	numerator = np.sum((fixed_norm - fixed_mean) * (moving_norm - moving_mean))
	denominator = np.sqrt(np.sum((fixed_norm - fixed_mean) ** 2) * np.sum((moving_norm - moving_mean) ** 2))
	ncc = numerator / denominator if denominator != 0 else 0

	# 3. Informação Mútua (MI)
	hist_2d, x_edges, y_edges = np.histogram2d(
		fixed_norm.ravel(),
		moving_norm.ravel(),
		bins=20
	)

	# Normaliza o histograma para obter probabilidades
	pxy = hist_2d / float(np.sum(hist_2d))
	px = np.sum(pxy, axis=1)
	py = np.sum(pxy, axis=0)
	px_py = px[:, None] * py[None, :]

	# Remove zeros para evitar log(0)
	nzs = pxy > 0
	mutual_info = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

	return {
		'MSE': mse,
		'NCC': ncc,
		'MI': mutual_info
	}

def analyze_registration_quality(reference_array, ID_image, registered_images_folder):
	"""
	Analisa a qualidade do registro entre uma imagem de referência e múltiplas imagens registradas

	Args:
		reference_image_path: Caminho para a imagem CT 3D de referência (A)
		registered_images_folder: Pasta contendo as 11 imagens registradas

	Returns:
		dict: Resultados da análise para cada imagem
	"""
	results = {}
	registered_files = sorted(Path(registered_images_folder).glob('*.npz'))  # Ajuste a extensão conforme necessário
	#print(f'Quantidade de imagens registradas encontradas na pasta: {len(registered_files)}')

	for reg_file in registered_files:
		moving_array = np.load(reg_file)["image"][:].astype(np.float32)
		moving_group = np.load(reg_file)["group"]
		moving_array = moving_array.transpose(2,1,0)

		#pickle_path = os.path.join('/mnt/data/Jean/registered_images_pickle/registered_images_lola11/groups/group_'+str(moving_group)+'/pickle/'+ID_image+'_'+str(moving_group)+'.pkl')

		#rigid_path = str(pickle_path)
		#with open(rigid_path, 'rb') as matrix_file:
		#	rigid = pickle.load(matrix_file)

		#reference_array = rigid.transform(image)

		reference_path = os.path.join(TEMP_IMAGES, 'registered_images/groups/group_'+str(moving_group), 'npz_rigid', ID_image+'.npz')
		reference_array = np.load(reference_path)["image"][:].astype(np.float32)
		reference_array = reference_array.transpose(2,1,0)

		#show_images(moving_array, reference_array)

		assert moving_array.shape==reference_array.shape, f'Shapes diferentes: {moving_array.shape} e {reference_array.shape}.'

		# Calcula as métricas
		metrics = calculate_similarity_metrics(reference_array, moving_array)
		results[reg_file.name] = metrics

	return results

def filter_dicom_list(path: List):
	initial_l = len(path)
	ps_dcms = [(p, pydicom.read_file(p)) for p in path]

	# Remove no slice location
	pre_len = len(ps_dcms)
	ps_dcms = [pydicom_tuple for pydicom_tuple in ps_dcms if hasattr(pydicom_tuple[1], "SliceLocation")]
	diff = pre_len - len(ps_dcms)
	if diff > 0:
		print(f"WARNING: Removed {diff} slices from series due to not having SliceLocation")

	# Order by slice location
	ps_dcms = sorted(ps_dcms, key=lambda s: s[1].SliceLocation)

	# Select most common shape
	ps_dcms_shapes = [(p, dcm, (dcm.Rows, dcm.Columns)) for p, dcm in ps_dcms]
	most_common_shape = Counter([shape for _, _, shape in ps_dcms_shapes]).most_common(1)[0][0]
	path = [(p, dcm) for p, dcm, dcm_shape in ps_dcms_shapes if dcm_shape == most_common_shape]
	path_diff = initial_l - len(path)
	if path_diff != 0:
		print(f"WARNING: {path_diff} slices removed due to misaligned shape.")
	ps_dcms = [(p, dcm) for p, dcm, _ in ps_dcms_shapes]

	# Select consistent SpacingBetweenSlices or SliceThickness
	initial_l = len(path)
	try:
		ps_dcms_spacings = [(p, dcm, dcm.SpacingBetweenSlices) for p, dcm in ps_dcms]
		attr = "spacing"
	except AttributeError:
		ps_dcms_spacings = [(p, dcm, dcm.SliceThickness) for p, dcm in ps_dcms]
		attr = "thickness"
	most_common_spacing = Counter([spacing for _, _, spacing in ps_dcms_spacings]).most_common(1)[0][0]
	path = [p for p, _, spacing in ps_dcms_spacings if spacing == most_common_spacing]
	path_diff = initial_l - len(path)
	if path_diff != 0:
		print(f"WARNING: {path_diff} slices removed due to inconsistent {attr}.")

	return path

def all_idx(idx, axis):
	grid = np.ogrid[tuple(map(slice, idx.shape))]
	grid.insert(axis, idx)
	return tuple(grid)

def int_to_onehot(matrix, overhide_max=None):
	'''
	Converts a matrix of int values (will try to convert) to one hot vectors
	'''
	if overhide_max is None:
		vec_len = int(matrix.max() + 1)
	else:
		vec_len = overhide_max

	onehot = np.zeros((vec_len,) + matrix.shape, dtype=int)

	int_matrix = matrix.astype(int)
	onehot[all_idx(int_matrix, axis=0)] = 1

	return onehot

def unified_img_reading(path, mask_path=None, lung_path=None, airway_path=None, torch_convert=False, isometric=False, convert_to_onehot=None, show_message=True):
	'''
	path: path to main image
	mask_path: path to segmentation image if available
	torch_convert: converts to torch format and expands channel dimension
	isometric: puts image to 1, 1, 1 voxel size

	returns: data, mask, spacing
	'''
	# Reorder by slice location and remove non aligned shapes
	if isinstance(path, list):
		filter_dicom_list(path)

	image = sitk.ReadImage(path)
	spacing = image.GetSpacing()[::-1]
	data = sitk.GetArrayFromImage(image)
	shape = data.shape
	directions = np.asarray(image.GetDirection())
	mask = None
	lung = None
	findings = None
	airway = None

	if len(directions) == 9:
		data = np.flip(data, np.where(directions[[0,4,8]][::-1]<0)[0]).copy()  # cryptic one liner from lungmask
		if mask_path is not None:
			mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
			mask = np.flip(mask, np.where(directions[[0,4,8]][::-1]<0)[0]).copy()
			mask = corrige_label(mask)
			print('Shape mask:', mask.shape)
		if lung_path is not None:
			lung = sitk.GetArrayFromImage(sitk.ReadImage(lung_path))
			lung[lung>1]=1
			lung = np.flip(lung, np.where(directions[[0,4,8]][::-1]<0)[0]).copy()
			print('Shape lung', lung.shape)
		if airway_path is not None:
			airway = sitk.GetArrayFromImage(sitk.ReadImage(airway_path))
			airway[airway>1]=1
			airway = np.flip(airway, np.where(directions[[0,4,8]][::-1]<0)[0]).copy()
			print('Shape airway', airway.shape)

	if isometric:
		if show_message:
			print(f"Pre isometry stats {spacing}: {data.shape} max {data.max()} min {data.min()}")
		data = zoom(data, spacing)
		if mask is not None:
			mask = zoom(mask, spacing, order=0)
			mask_shape = mask.shape
		else:
			mask_shape = None
		if lung is not None:
			lung = zoom(lung, spacing, order=0)
			lung_shape = lung.shape
		else:
			lung_shape = None
		if airway is not None:
			airway = zoom(airway, spacing, order=0)
			airway_shape = airway.shape
		else:
			airway_shape = None
		if show_message:
			print(f"Post isometry stats {spacing} ->~ [1, 1, 1]: {data.shape}/{mask_shape}{lung_shape}{airway_shape} max {data.max()} min {data.min()}")
		spacing = [1.0, 1.0, 1.0]

	if torch_convert:
		if data.dtype == "uint16":
			data = data.astype(np.int16)
		data = torch.from_numpy(data).float()
		if len(data.shape) == 3:
			data = data.unsqueeze(0)
		if mask_path is not None:
			if mask.dtype == "uint16":
				mask = mask.astype(np.int16)
			if convert_to_onehot is not None:
				if len(mask.shape) > 3:
					raise ValueError(f"Are you sure you want to convert to one hot a mask that is of this shape: {mask.shape}")
				mask = int_to_onehot(mask, overhide_max=convert_to_onehot)
			mask = torch.from_numpy(mask).float()
		if lung_path is not None:
			if lung.dtype == "uint16":
				lung = lung.astype(np.int16)
			#if convert_to_onehot is not None:
			#	if len(lung.shape) > 3:
			#		raise ValueError(f"Are you sure you want to convert to one hot a mask that is of this shape: {mask.shape}")
			#	lung = int_to_onehot(lung, overhide_max=convert_to_onehot)
			lung = torch.from_numpy(lung).float()
		if airway_path is not None:
			if airway.dtype == "uint16":
				airway = airway.astype(np.int16)
			#if convert_to_onehot is not None:
			#	if len(lung.shape) > 3:
			#		raise ValueError(f"Are you sure you want to convert to one hot a mask that is of this shape: {mask.shape}")
			#	lung = int_to_onehot(lung, overhide_max=convert_to_onehot)
			airway = torch.from_numpy(airway).float()

	return data, mask, lung, airway, spacing, shape

def load_config(config_name):
	with open(os.path.join(config_name)) as file:
		config = yaml.safe_load(file)

	return config

def busca_path(ID_image, group):

	rigid_path = os.path.join(TEMP_IMAGES, 'registered_images/groups/group_'+str(group)+'/pickle', ID_image+'_'+str(group)+'.pkl')

	return rigid_path

def teste_pickle_by_image(ID_image, group=None):
	if group is None:
		for group in range(1,12):
			pickle_path = os.path.join(TEMP_IMAGES, 'registered_images/groups/group_'+str(group)+'/pickle/'+ID_image+'_'+str(group)+'.pkl')

			if os.path.exists(pickle_path)==False:
				print(f'Pickle não existe: {ID_image}')
				return False
			else:
				rigid_path = str(pickle_path)
				with open(rigid_path, 'rb') as matrix_file:
					rigid = pickle.load(matrix_file)
	else:
		pickle_path = os.path.join(TEMP_IMAGES, 'registered_images/groups/group_'+str(group)+'/pickle/'+ID_image+'_'+str(group)+'.pkl')

		if os.path.exists(pickle_path)==False:
			print(f'Pickle não existe: {ID_image}')
			return False
		else:
			rigid_path = str(pickle_path)
			with open(rigid_path, 'rb') as matrix_file:
				rigid = pickle.load(matrix_file)

	return True

def register_single(moving_path, moving_label_path, moving_lung_path, moving_airway_path, group):
	GROUP = group

	OUTPUT_DIR = os.path.join(TEMP_IMAGES, "registered_images/groups/group_"+str(GROUP))

	OUTPUT_DIR_PICKLE = os.path.join(OUTPUT_DIR, 'pickle')
	OUTPUT_DIR_NPZ_RIGID = os.path.join(OUTPUT_DIR, 'npz_rigid')

	os.makedirs(OUTPUT_DIR_PICKLE, exist_ok=True)
	os.makedirs(OUTPUT_DIR_NPZ_RIGID, exist_ok=True)

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

	#fixed_label_path = os.path.join(RAW_DATA_FOLDER, 'labels', ID_fixed+'.nii.gz')

	template_img = nib.load(fixed_path)
	#template_label_img = nib.load(fixed_label_path)

	template_data = template_img.get_fdata()
	template_grid2world = template_img.affine
	#template_label_data = template_label_img.get_fdata()

	if ID_moving != ID_fixed:
		moving_img = nib.load(moving_path)
		if moving_label_path is not None:
			moving_label = nib.load(moving_label_path)
			moving_label_data = moving_label.get_fdata()
		if moving_lung_path is not None:
			moving_lung = nib.load(moving_lung_path)
			moving_lung_data = moving_lung.get_fdata()
		if moving_airway_path is not None:
			moving_airway = nib.load(moving_airway_path)
			moving_airway_data = moving_airway.get_fdata()

		moving_data = moving_img.get_fdata()
		moving_grid2world = moving_img.affine

		print(moving_data.shape, template_data.shape)




		c_of_mass = transform_centers_of_mass(template_data, template_grid2world, moving_data, moving_grid2world)

		print('Transform centers of mass done!')

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

		print('TranslationTransform3D done!')

		transform = RigidTransform3D()
		params0 = None
		starting_affine = translation.affine
		rigid = affreg.optimize(template_data, moving_data, transform, params0, template_grid2world, moving_grid2world, starting_affine=starting_affine)

		transformed = rigid.transform(moving_data)
		if moving_label_path:
			transformed_label = rigid.transform(moving_label_data, interpolation='nearest')
		if moving_lung_path is not None:
			transformed_lung = rigid.transform(moving_lung_data, interpolation='nearest')
		if moving_airway_path is not None:
			transformed_airway = rigid.transform(moving_airway_data, interpolation='nearest')

		pickle_path = os.path.join(OUTPUT_DIR_PICKLE, f'{ID_moving}_{group}.pkl')
		with open(pickle_path, 'wb') as matrix_file:
			pickle.dump(rigid, matrix_file)

		save_path = os.path.join(OUTPUT_DIR_NPZ_RIGID, f'{ID_moving}.npz')
		if moving_label_path is not None:
			np.savez_compressed(save_path, image=transformed, label=transformed_label, lung=transformed_lung, airway=transformed_airway, group=group, ID=ID_moving, pickle_path=pickle_path)
		else:
			np.savez_compressed(save_path, image=transformed, group=group, ID=ID_moving, pickle_path=pickle_path)

		print('RigidTransform3D done!')

	return None

def random_crop_ZXY(image, lung, label, airway, crop_size=(128,128,128)):
	depth, width, height = crop_size[0], crop_size[1], crop_size[2]

	assert (len(image.shape)==5 and len(lung.shape)==5 and len(label.shape)==5 and len(airway.shape)==5), f'Tamanho de shape diferente de 5.'
	assert image.shape[3] == image.shape[4], f'{image.shape}'

	if (image.shape[2] < depth):
		depth = image.shape[2]
	assert image.shape[4] >= depth
	assert image.shape[2] >= height
	assert image.shape[3] >= width

	z = random.randint(0, image.shape[2] - depth)
	x = random.randint(0, image.shape[3] - height)
	y = random.randint(0, image.shape[4] - width)

	image = image[:, :, z:z+depth, x:x+height, y:y+width]
	lung = lung[:, :, z:z+depth, x:x+height, y:y+width]
	label = label[:, :, z:z+depth, x:x+height, y:y+width]
	airway = airway[:, :, z:z+depth, x:x+height, y:y+width]

	return image, lung, label, airway

def random_crop_XYZ(image, lung, label, airway, crop_size=(128,128,128)):
	depth, width, height = crop_size[0], crop_size[1], crop_size[2]

	assert (len(image.shape)==5 and len(lung.shape)==5 and len(label.shape)==5 and len(airway.shape)==5), f'Tamanho de shape diferente de 5.'
	assert image.shape[2] == image.shape[3], f'{image.shape}'

	if (image.shape[4] < depth):
		depth = image.shape[4]
	assert image.shape[4] >= depth
	assert image.shape[2] >= height
	assert image.shape[3] >= width

	z = random.randint(0, image.shape[4] - depth)
	x = random.randint(0, image.shape[2] - height)
	y = random.randint(0, image.shape[3] - width)

	image = image[:, :, x:x+height, y:y+width, z:z+depth]
	lung = lung[:, :, z:z+depth, x:x+height, y:y+width]
	label = label[:, :, x:x+height, y:y+width, z:z+depth]
	airway = airway[:, :, x:x+height, y:y+width, z:z+depth]

	return image, lung, label, airway

def random_crop(image, lung, label, airway, crop_size=(128,128,128)):
	if image.shape[2]==image.shape[3]:
		return random_crop_XYZ(image, lung, label, airway, crop_size=crop_size)
	elif image.shape[3]==image.shape[4]:
		return random_crop_ZXY(image, lung, label, airway, crop_size=crop_size)
