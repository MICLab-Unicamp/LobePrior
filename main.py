#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import torch
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import NeptuneLogger

from dataModule import DataModule
from lightning_7_decoders import Lightning
from utils.general import load_config

MODES = ["train", "val"]

def train(hparams):
	# Passe a API KEY do Neptune
	api_key = os.getenv("NEPTUNE_API_TOKEN")

	debug = hparams.debug
	neptune_offline = hparams.neptune_offline

	# Inicialização do LightningModule
	model = Lightning(hparams)

	tags = model.hparams.approach, model.hparams.loss
	path_check_point_lobes = hparams.experiment_name + "_{epoch}-{val_loss:.3f}_"+hparams.approach+"_lr="+str(hparams.lr)+"_"+hparams.optimizer+'_'+hparams.loss+'_saida='+str(hparams.snout)+'_lobes'
	path_check_point_airway = hparams.experiment_name + "_{epoch}-{val_loss:.3f}_"+hparams.approach+"_lr="+str(hparams.lr)+"_"+hparams.optimizer+'_'+hparams.loss+'_saida='+str(hparams.snout)+'_airway'

	data = DataModule(hparams)

	# Recomenda-se não inicializar o logger se estiver fazendo debug
	if debug:
		neptune_logger = None
		logger = None
		checkpoint_callback = None

		#data.check_dataset()
	else:
		# Onde salvar checkpoints
		try:
			ckpt_path = "logs_3D/"
			os.makedirs(ckpt_path, exist_ok=True)
			print(f'Directory {ckpt_path} created successfully!')
		except OSError as error:
			ckpt_path = "logs/"
			os.makedirs(ckpt_path, exist_ok=True)
			print(f'Local directory {ckpt_path} can not be created!')

		if neptune_offline==True:
			print('Neptune offline: utilizando TensorBoardLogger.')

			from pytorch_lightning.loggers import TensorBoardLogger

			logger = TensorBoardLogger("tb_logs", name="CBEB_Lesions")
		else:
			# Objeto do logger, neste caso o Neptune.
			neptune_logger = NeptuneLogger(
				api_key=api_key,
				project="jeankusanagi/sandbox",
				log_model_checkpoints=False,
				tags=tags,
				)
			neptune_logger.log_hyperparams(model.hparams)

		early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.0, patience=model.hparams.patience, verbose=False, mode="min")
		model_checkpoint_lobes = ModelCheckpoint(filename=path_check_point_lobes,
													dirpath=ckpt_path,  # path onde será salvo o checkpoint
													monitor="val_loss_lobes",
													mode="min",
													)
		model_checkpoint_airway = ModelCheckpoint(filename=path_check_point_airway,
													dirpath=ckpt_path,  # path onde será salvo o checkpoint
													monitor="val_loss_aiway",
													mode="min",
													)

		# Configuração do Checkpoint
		# Importante prestar atenção nas opções do ModelCheckpoint para não acabar dando overwrite em pesos antigos sem querer!
		if (hparams.early_stop_callback):
			checkpoint_callback = [model_checkpoint_lobes, early_stop_callback]
		else:
			checkpoint_callback = [model_checkpoint_lobes]

	trainer = pl.Trainer(
						max_epochs=hparams.max_epochs,
						devices=1 if torch.cuda.is_available() else 0,
						fast_dev_run=debug,
						logger=logger if hparams.neptune_offline else neptune_logger,
						callbacks=checkpoint_callback,
						accelerator="gpu",
						#accumulate_grad_batches=2
						)

	trainer.fit(model, data)

def main(cli_args):
	print('Parâmetros:', cli_args)

	parser = argparse.ArgumentParser(description='Lung lobe segmentation on CT images using Priori.')
	parser.add_argument('--experiment_name', default="LightningLobes_OneNetwork", type=str)
	parser.add_argument('--approach', default='attUnet_template', type=str)
	parser.add_argument('--batch_size', default=1, type=int)
	parser.add_argument('--dataset_path', default=None, type=str)
	parser.add_argument('--max_epochs', type=int, default=1000, help= "Number of epochs (int).")
	parser.add_argument('--loss', default='focal_loss_kaggle', type=str)
	parser.add_argument('--lr', default=1e-4, type=float)
	parser.add_argument('--optimizer', default='AdamW', type=str)
	parser.add_argument('--scheduler', default=None, type=str)
	parser.add_argument('--nin',  default=1, type=int, help="Number of input channels")
	parser.add_argument('--snout', default=6, type=int)
	parser.add_argument('--weight_decay', default=1e-5, type=float)
	parser.add_argument('--patience', default=100, type=int)
	parser.add_argument('--spacing', default='0', type=str)
	parser.add_argument('--spacing_voxel', default=None, help='Original resolution')
	parser.add_argument('--crop_size', default=(128, 256, 256))
	parser.add_argument('--roi_size_train', default=(128, 256, 256))
	parser.add_argument('--roi_size_val', default=(128, 128, 128))
	parser.add_argument('--train_transform_str', default=None)
	parser.add_argument('--eval_transform_str', default=None)
	parser.add_argument('--mode', default="segmentation", type=str)
	parser.add_argument('--datatype', default="npz", type=str)
	parser.add_argument('--nworkers', default=4, help="Number of workers", type=int)

	parser.add_argument('--use_checkpoint', action="store_true", help="Use gradient checkpointing to save memory") 	# true se passou --use_checkpoint
	parser.add_argument('--early_stop_callback', action="store_true") 		# true se passou --early_stop_callback
	parser.add_argument('--neptune_offline', action="store_true")			# true se passou --neptune_offline
	parser.add_argument('--salve_image', action="store_true")				# true se passou --salve_image
	parser.add_argument('--config', action="store_true") 					# true se passou --debug
	parser.add_argument('--debug', action="store_true") 					# true se passou --debug

	args = parser.parse_args()
	args.cli_args = ' '.join(cli_args)
	args.approachs_used_3d = ['vnet3d_monai','unet3d','unet3d_diedre','attentionUnet3D','residual_unet','munet3d']

	if args.config:
		config = load_config("my_config.yaml")

		args.experiment_name = config['experiment_name']
		args.max_epochs = config['max_epochs']
		args.snout = config['snout']
		args.patience = config['patience']
		args.nworkers = config['train']['nworkers']
		args.crop_size = config['train']['roi_size_train']
		args.roi_size_train = config['train']['roi_size_train']
		args.roi_size_val = config['train']['roi_size_val']
		args.train_transform_str = config['train']['train_transform_str']
		args.roi_size = config['val']['roi_size_val']
		args.eval_transform_str = config['val']['eval_transform_str']

	parser.print_help()

	print('\nArgs:')
	for k, v in sorted(vars(args).items()):
		print(f'\t{k}: {v}')

	#for arg in vars(args):
	#	print(f'{arg}: {getattr(args, arg)}')
	print('\n')

	train(args)

if __name__ == '__main__':
	os.system('cls' if os.name == 'nt' else 'clear')

	print(f"PyTorch version: {torch.__version__}")
	print(f"Pytorch Lightning version: {pl.__version__}\n")

	sys.exit(main(sys.argv))
