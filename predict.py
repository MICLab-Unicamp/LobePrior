#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import monai
import torch
import skimage
import numpy as np
import torchio as tio
import SimpleITK as sitk
import pytorch_lightning as pl
from lungmask import mask as lungmask
from matplotlib import pyplot as plt
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

from utils.general import SalvaFile
from model.unet_diedre import UNet_Diedre
from utils.post_processed_3D import pos_processed
from utils.to_onehot import to_onehot
from utils.metric import Dice_chavg_per_label_metric
from utils.loss import FocalLossKaggle, BayesianLossByChannel
from utils.metric import Dice_metric
from utils.post_processed_3D import pos_processed
from utils.seg_metrics import initialize_metrics_dict, seg_metrics
from utils.unified_img_reading_resample import unified_img_reading

RAW_DATA_FOLDER = os.getenv("HOME")
DATA_FOLDER_NPZ = 'output_predict'

def show_output(image, epoch=0, msg=None):
	#print(image.shape)

	if msg==None:
		msg='Segmented Image'

	if image.shape[1]==5:
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
	elif image.shape[1]==6:
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
		#save_path = os.path.join('output_'+str(epoch)+'.png')
		#plt.savefig(save_path, dpi=300, transparent=True)
		plt.show()
		plt.close()

def pos_processamento(output, template, segmentation=None):

	output = output * segmentation  # B, C, Z, Y, X * B, 1, Z, Y, X

	template = template.argmax(dim=1)
	output = output.argmax(dim=1)
	segmentation = segmentation[:, 0]
	output = torch.where((segmentation == 1)*(output == 0), template, output)

	return output, template, segmentation

class LoberModule(pl.LightningModule):
	def __init__(self, hparams):
		super().__init__()

		# O nome precisar ser hparams para o PL.
		self.save_hyperparameters(hparams)

		# Loss
		#self.criterion_seg = FocalLossKaggle()
		self.criterion_baysian = BayesianLossByChannel()
		self.criterion_by_lobe = monai.losses.DiceLoss(reduction='mean')

		# Métricas
		self.metric = Dice_chavg_per_label_metric()

		if self.hparams.mode == "segmentation":
			#self.model_seg = BB_Unet(n_organs=1, BB_boxes=1)
			#self.model_seg = BB_Unet_pytorch(in_channels=1, n_classes=1, s_channels=64, BB_boxes=1)
			#self.model_seg = UNet_Diedre(n_channels=3, n_classes=1, norm=True, dim='3d', init_channel=16, joany_conv=False, dict_return=False)
			self.model_hat = UNet_Diedre(n_channels=1, n_classes=6, norm="instance", dim='3d', init_channel=16, joany_conv=False, dict_return=False)
			self.model_seg_0 = UNet_Diedre(n_channels=3, n_classes=1, norm="instance", dim='3d', init_channel=8, joany_conv=False, dict_return=False)
			self.model_seg_1 = UNet_Diedre(n_channels=3, n_classes=1, norm="instance", dim='3d', init_channel=8, joany_conv=False, dict_return=False)
			self.model_seg_2 = UNet_Diedre(n_channels=3, n_classes=1, norm="instance", dim='3d', init_channel=8, joany_conv=False, dict_return=False)
			self.model_seg_3 = UNet_Diedre(n_channels=3, n_classes=1, norm="instance", dim='3d', init_channel=8, joany_conv=False, dict_return=False)
			self.model_seg_4 = UNet_Diedre(n_channels=3, n_classes=1, norm="instance", dim='3d', init_channel=8, joany_conv=False, dict_return=False)
			#self.model_hat = monai.networks.nets.VNet(in_channels = 6, out_channels = self.hparams.snout)

	def forward_per_lobe(self, x, template, y_seg, y):
		#print(x.shape, y.shape, template.shape)

		template = (template > 0.3).float()

		lobes_regions = skimage.measure.regionprops(np.array(template.squeeze().cpu().argmax(axis=0)).astype(np.uint8))

		buffer = []
		dice_cor = []

		#show_image(y, template, y_seg)

		for i in range(5):
			lobe_region = lobes_regions[i].bbox

			# Make lobe region a 1 cube
			box = np.zeros((1, 1) + template[0,0].shape, dtype=np.uint8)
			box[:,:,lobe_region[0]:lobe_region[3], lobe_region[1]:lobe_region[4], lobe_region[2]:lobe_region[5]] = 1

			#label = y[:,i+1:i+2]
			label_seg = y_seg[:,i+1:i+2]
			#print(f'{i} -> {label_seg.shape}')

			# Concatenate cube with CT
			box = torch.from_numpy(box).cuda()

			#print(x.shape, label_seg.shape, box.shape)
			x_new = torch.cat((x, label_seg, box), dim = 1)

			#show_box(x_new.cpu(), y_seg.cpu(), template.cpu(), y.cpu(), box.cpu(), i)

			if i==0:
				model = self.model_seg_0
			elif i==1:
				model = self.model_seg_1
			elif i==2:
				model = self.model_seg_2
			elif i==3:
				model = self.model_seg_3
			elif i==4:
				model = self.model_seg_4
			y_new = y

			output = sliding_window_inference(
				x_new.cuda(),
				roi_size=(128, 128, 128),
				sw_batch_size=1,
				predictor=model.cuda(),
				overlap=0.3,
				mode="gaussian",
				progress=False,
				device=torch.device('cuda')
			)
			output = output.sigmoid()
			buffer.append(output)

			dice_cor.append(Dice_metric(output, y_new[:,i+1:i+2]).mean())

		# Concatenate on channel dimension
		y_hat_seg = torch.cat(buffer, dim=1)

		return y_hat_seg, dice_cor

	def forward_seg(self, x):
		y_hat = self.model_hat(x)

		return y_hat.softmax(dim=1)

	def forward(self, x_high, x, template, y):
		y_seg = self.forward_seg(x_high)
		y_seg_resize = torch.nn.functional.interpolate(y_seg.detach(), size=x[0,0].shape, mode='nearest')

		output, dice_cor = self.forward_per_lobe(x, template, y_seg_resize, y)
		return y_seg, output, dice_cor

	def validation_step(self, test_batch):
		x_high, y_high, template_high, x, y, template = test_batch["image_h"], test_batch["label_h"], test_batch["template_high"], test_batch["image"], test_batch["label"], test_batch["template"]
		y_seg, output, dice_cor = self.forward(x_high, x, template, y)

		return output.cpu(), dice_cor

	def predict(self, all_images, checkpoint, post_processed=True, save_image=False, dataset_name='coronacases') -> np.ndarray:

		#all_images =['/home/jean/outputs_images_and_labels/group_11/npz_rigid/3_rigid3D.npz']
		#all_images =['/home/jean/outputs_images_and_labels/group_8/npz_rigid/coronacases_003_rigid3D.npz']
		#all_images =['/home/jean/outputs_images_and_labels/group_8/npz_rigid/coronacases_010_rigid3D.npz']

		#TEMPLATE_8_PATH = os.path.join(RAW_DATA_FOLDER, 'DataSets/outputs_registered_high/isometric_cliped_and_normalized/model_fusion/group_8.npz')
		#npz_template_path = TEMPLATE_8_PATH
		#TEMPLATE_11_PATH = os.path.join(RAW_DATA_FOLDER, 'DataSets/outputs_registered_high/isometric_cliped_and_normalized/model_fusion/group_11.npz')
		#npz_template_path = TEMPLATE_11_PATH

		#npz_path = all_images[0]
		print('Quantidades de imagens encontradas no dataset:', len(all_images))

		npz_template_path = None
		if (save_image):
			ckpt_path = 'results/outputs'
			os.makedirs(ckpt_path, exist_ok=True)

		if dataset_name is None:
			arq = SalvaFile(resutls_path='results/save_lobes_pos_reforcado_3D_')
		else:
			arq = SalvaFile(resutls_path='results/save_lobes_pos_reforcado_3D_'+dataset_name+'_'+str(checkpoint['epoch']))

		struct_names=['BG','LUL','LLL','RUL','RML','RLL']

		for instance, npz_path in enumerate(all_images):
			print(f'\nInstance {instance+1}')

			ID_image = os.path.basename(npz_path).replace('.npz','').replace('_affine3D','').replace('_rigid3D','')
			print(f'\tImage name: {ID_image}')

			if npz_template_path is None:
				for group in range(11):
					template_path = os.path.join(RAW_DATA_FOLDER, 'DataSets/outputs_registered_high/isometric_cliped_and_normalized/groups', 'group_'+str(group+1), 'npz_rigid', ID_image+'.npz')
					if os.path.isfile(template_path):
						npz_template_path = os.path.join(RAW_DATA_FOLDER, 'DataSets/outputs_registered_high/isometric_cliped_and_normalized/model_fusion/group_'+str(group+1)+'.npz')
						print('\tTemplate path:', npz_template_path)

			npz = np.load(npz_path)
			img, tgt = npz["image"][:].astype(np.float32), npz["label"][:].astype(np.float32)
			template = np.load(npz_template_path)["model"][:].astype(np.float32)

			img = img.transpose(2,1,0)
			tgt = tgt.transpose(2,1,0)

			if len(tgt.shape)==3:
				if tgt.max()==8 or tgt.max()==520:
					mask_one, onehot, labels_one = to_onehot(tgt, [7,8,4,5,6], single_foregound_lable=False, onehot_type=np.dtype(np.int8))
				elif tgt.max()==5:
					mask_one, onehot, labels_one = to_onehot(tgt, [1,2,3,4,5], single_foregound_lable=False, onehot_type=np.dtype(np.int8))

				tgt = onehot
				img = np.expand_dims(img, 0)

			subject = tio.Subject(
				image=tio.ScalarImage(tensor = img),
				label=tio.LabelMap(tensor = tgt),
				template=tio.LabelMap(tensor = template),
			)
			transform = tio.Resize((128, 128, 128))
			transformed = transform(subject)
			img_high = transformed.image.numpy()
			tgt_high = transformed.label.numpy()
			template_high = transformed.template.numpy()

			#assert template_high.shape == tgt_high.shape, 'Template and label should be same shape, instead are {}, {}'.format(template.shape, tgt.shape)
			#assert len(template.shape) == 4, 'Inputs should be B*W*H*N tensors, instead have shape {}'.format(img.shape)

			#showImages(img_high, tgt_high)
			#showImages(img, tgt)
			#showImages(img, template)

			segmentation = np.array(tgt.squeeze().argmax(axis=0)).astype(np.uint8)
			if segmentation.max()>8:
				segmentation[segmentation>8]=0
			segmentation[segmentation>0]=1

			new_image = np.zeros(img[0].shape).astype(img.dtype)
			new_image = np.where(segmentation == 1, img, img.min())
			img = new_image

			img_high = torch.tensor(img_high, dtype=torch.float32).unsqueeze(dim=0).cuda()
			tgt_high = torch.tensor(tgt_high, dtype=torch.float32).unsqueeze(dim=0).cuda()
			template = torch.tensor(template, dtype=torch.float32).unsqueeze(dim=0).cuda()
			img = torch.tensor(img, dtype=torch.float32).unsqueeze(dim=0).cuda()
			tgt = torch.tensor(tgt, dtype=torch.float32).unsqueeze(dim=0).cuda()
			template_high = torch.tensor(template_high, dtype=torch.float32).unsqueeze(dim=0).cuda()
			segmentation = torch.from_numpy(segmentation).unsqueeze(dim=0).unsqueeze(dim=0).float()

			print('\tShape high:', img_high.shape, tgt_high.shape, template_high.shape)
			print('\tShape:', img.shape, tgt.shape, template.shape, segmentation.shape)
			print('\tMinMax x:', img.min(), img.max())
			print('\tMinMax y:', tgt.min(), tgt.max())
			print('\tMinMax template:', template.min(), template.max())

			assert len(img_high.shape)==len(tgt_high.shape)==len(template_high.shape)==len(img.shape)==len(tgt.shape)==len(template.shape)==len(segmentation.shape)
			assert tgt_high.shape==template_high.shape
			assert tgt.shape==template.shape

			sample = {"image_h": img_high, "label_h": tgt_high, 'template_high': template_high, "image": img, "label": tgt, "template": template, "npz_path": npz_path, "ID":ID_image}

			self.eval()
			with torch.no_grad():
				y_hat_seg, dice_cor = self.validation_step(sample)

			#show_output(y_hat_seg.cpu(), msg='Segmented image without background')
			print(f'Shape output: {y_hat_seg.shape}')
			print(f'Dice: {dice_cor}')
			print(f'Dice mean: {torch.stack(dice_cor).mean()}')

			dice_without_backgound = Dice_metric(y_hat_seg.cuda(), tgt[:,1:6]).mean()
			print('Dice value before post-processing (image without backgound):', dice_without_backgound)

			################################################################################################
			lung = y_hat_seg.sum(dim=1).squeeze()
			lung_votes = (lung - lung.min())/(lung.max() - lung.min())
			lung_array = np.array(lung_votes.numpy()).astype(np.float32)
			bg_heatmap = 1 - torch.clip(lung, 0, 1)
			image = torch.cat([bg_heatmap.unsqueeze(0), y_hat_seg[0]], dim=0)

			#image = torch.zeros((1,6) + y_hat_seg[0,0].shape, dtype=y_hat_seg.dtype)
			#image[:,0:1] = 1-segmentation[:,0:1]
			#image[:,1:6] = y_hat_seg
			################################################################################################

			#show_output(image.unsqueeze(0), msg='Segmented image with background')

			if (save_image):
				output = image.squeeze().argmax(dim=0)
				output = sitk.GetImageFromArray(output.numpy().astype(np.uint8))
				sitk.WriteImage(output, os.path.join(ckpt_path, ID_image+"_sem_processamento.nii.gz"))

			print('Shape', image.shape, img.shape, tgt.shape, template.shape, segmentation.shape)

			if post_processed:
				image = image.unsqueeze(0)
				image = pos_processed(image)

				image, template, segmentation = pos_processamento(output=image.cpu(), template=template.cpu(), segmentation=segmentation)

				if (save_image):
					#output = sitk.GetImageFromArray(image.squeeze().argmax(dim=0).numpy())
					#sitk.WriteImage(output, os.path.join(ckpt_path, ID_image+"_pos_processed.nii.gz"))
					#output = sitk.GetImageFromArray(segmentation.squeeze().numpy())
					#sitk.WriteImage(output, os.path.join(ckpt_path, ID_image+"_segmentation.nii.gz"))
					#output = sitk.GetImageFromArray(template.squeeze().numpy())
					#sitk.WriteImage(output, os.path.join(ckpt_path, ID_image+"_template.nii.gz"))
					output = sitk.GetImageFromArray(image.squeeze().numpy())
					sitk.WriteImage(output, os.path.join(ckpt_path, ID_image+"_pos_processamento.nii.gz"))

				#show_output(image, msg='Segmented image with background after post-processing')

				image = image.squeeze().numpy()
				if len(image.shape)==3:
					if image.max()==8 or image.max()==520:
						mask_one, onehot, labels_one = to_onehot(image, [7,8,4,5,6], single_foregound_lable=False, onehot_type=np.dtype(np.int8))
					elif image.max()==5:
						mask_one, onehot, labels_one = to_onehot(image, [1,2,3,4,5], single_foregound_lable=False, onehot_type=np.dtype(np.int8))
					image = onehot
				else:
					print('Shape errado')
				image = torch.tensor(image, dtype=torch.float32).unsqueeze(dim=0)
				print('Shape:', image.shape, tgt.shape)

				dice = Dice_metric(image.cuda(), tgt).mean()
				print('Dice value after post-processing (image with backgound):', dice)
				dice = Dice_metric(image[:,1:6].cuda(), tgt[:,1:6]).mean()
				print('Dice value after post-processing (image without backgound):', dice)

				dsc_seg, dsc_lobes_seg = self.metric(image.cuda(), tgt)
				print(dsc_seg)
				print(dsc_lobes_seg)

				metrics = initialize_metrics_dict()
				seg_metrics(gts=tgt[0].cpu().numpy().astype(np.uint8), preds=image[0].cpu().numpy().astype(np.uint8), metrics=metrics, struct_names=struct_names)
				arq.salva_arq(metrics, struct_names=struct_names)
				print(f'Fim da instancia {instance+1}')

			npz_template_path = None

def main(args):
	print('Parâmetros:', sys.argv[1])
	pre_trained_model_path = sys.argv[1]

	test_model = LoberModule.load_from_checkpoint(pre_trained_model_path)
	checkpoint = torch.load(pre_trained_model_path, map_location=lambda storage, loc: storage)

	#all_images = sorted(glob.glob(os.path.join(RAW_DATA_FOLDER, 'DataSets/outputs_registered_high/isometric_cliped_and_normalized/npz_rigid/test/luna16','*.npz')))
	#test_model.predict(all_images, checkpoint, post_processed=True, save_image=False, dataset_name='luna16')

	all_images = sorted(glob.glob(os.path.join(RAW_DATA_FOLDER, 'DataSets/outputs_registered_high/isometric_cliped_and_normalized/npz_rigid/test/coronacases','*.npz')))
	test_model.predict(all_images, checkpoint, post_processed=True, save_image=False, dataset_name='coronacases')

	return 0

if __name__ == '__main__':
	os.system('cls' if os.name == 'nt' else 'clear')
	sys.exit(main(sys.argv))
