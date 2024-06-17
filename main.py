#!/usr/bin/env python
# -*- coding: utf-8 -*-

#python main.py --approach vnet3d_monai --batch_size 1 --loss focal_loss_kaggle --max_epochs 3000 --early_stop_callback --patience 500 --lr_scheduler CosineAnnealingLR --nworkers 1 --debug

import os
import sys
import argparse
import torch
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import NeptuneLogger

from dataModule import DataModule
from lightning import Lightning

MODES = ["train", "val"]

def train(hparams):
	# Passe a API KEY do Neptune
	api_key = os.getenv("NEPTUNE_API_TOKEN")

	debug = hparams.debug
	neptune_offline = hparams.neptune_offline

	# Inicialização do LightningModule
	model = Lightning(hparams)

	if (any(x==hparams.approach for x in hparams.approachs_used_3d)):
		tags = model.hparams.approach, model.hparams.loss
		path_check_point = hparams.experiment_name + "_{epoch}-{val_loss:.3f}_"+hparams.approach+"_lr="+str(hparams.lr)+"_"+hparams.optimizer+'_'+hparams.loss+'_saida='+str(hparams.snout)

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
			ckpt_path = "logs_3D/byLobes_3D_Final/"
			os.makedirs(ckpt_path, exist_ok=True)
			print(f'Directory "{ckpt_path}" created successfully!')
		except OSError as error:
			ckpt_path = "logs/"
			os.makedirs(ckpt_path, exist_ok=True)
			print(f'Local directory "{ckpt_path}" can not be created!')

		if neptune_offline==True:
			print('Neptune offline: utilizando TensorBoardLogger.')
			#import neptune.new as neptune
			#run = neptune.init_run(mode="offline",project="jeankusanagi/sandbox",tags=tags)
			#neptune_logger = NeptuneLogger(run=run)

			from pytorch_lightning.loggers import TensorBoardLogger

			logger = TensorBoardLogger("tb_logs", name="model_by_lobes_3D_Dice")
		else:
			# Objeto do logger, neste caso o Neptune.
			# A API Key deve ser secreta e você consegue criando uma conta no Neptune.
			neptune_logger = NeptuneLogger(
				api_key=api_key,
				project="jeankusanagi/sandbox",
				log_model_checkpoints=False,
				tags=tags,
				)
			neptune_logger.log_hyperparams(model.hparams)

		early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.0, patience=model.hparams.patience, verbose=False, mode="min")
		model_checkpoint_callback = ModelCheckpoint(filename=path_check_point,
													dirpath=ckpt_path,  # path onde será salvo o checkpoint
													monitor="val_loss",
													mode="min",
													)

		# Configuração do Checkpoint
		# Importante prestar atenção nas opções do ModelCheckpoint para não acabar dando overwrite em pesos antigos sem querer!
		if (hparams.early_stop_callback):
			checkpoint_callback = [model_checkpoint_callback, early_stop_callback]
		else:
			checkpoint_callback = [model_checkpoint_callback]

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

	parser = argparse.ArgumentParser(description='Lung lobe segmentation on CT images.')
	parser.add_argument('--experiment_name', default="LightningByLobes_Final", type=str)
	#parser.add_argument('--approach', "-a", default=None, required=True, type=str)
	parser.add_argument('--batch_size', "-b", default=1, type=int)
	parser.add_argument('--dataset_path', "-ds", default=None, type=str)
	parser.add_argument('--max_epochs', "-me", type=int, default= 1000, help= "Number of epochs (int).")
	#parser.add_argument('--loss', "-l", default='focal_loss_kaggle', type=str)
	parser.add_argument('--lr', "-lr", default=1e-4, type=float)
	parser.add_argument('--optimizer', "-o", default='AdamW', type=str)
	parser.add_argument('--scheduler', "-lrs", default=None, type=str)
	parser.add_argument('--nin', "-n", default=1, type=int, help="Number of input channels")
	parser.add_argument('--snout', "-s", default=6, type=int)
	parser.add_argument('--weight_decay', "-wd", default=1e-5, type=float)
	parser.add_argument('--patience', "-p", default=100, type=int)
	parser.add_argument('--n_splits', "-ns", default=5, type=int)
	parser.add_argument('--plano', "-pl", default='axial', type=str)
	parser.add_argument('--spacing', "-sp", default='0', type=str)
	parser.add_argument('--spacing_voxel', default=None, help='Original resolution')
	parser.add_argument('--shape_random_crop', default=(128, 256, 256))
	parser.add_argument('--train_transform_str', "-tt", default=None)
	parser.add_argument('--eval_transform_str', "-et", default=None)
	parser.add_argument('--mode', default="segmentation", type=str)
	parser.add_argument('--datatype', default="npz", type=str)
	parser.add_argument('--nworkers', "-nw", default=4, help="Number of workers", type=int)

	parser.add_argument('--use_checkpoint', action="store_true", help="Use gradient checkpointing to save memory") 	# true se passou --use_checkpoint
	parser.add_argument('--early_stop_callback', action="store_true") 		# true se passou --early_stop_callback
	parser.add_argument('--neptune_offline', action="store_true")			# true se passou --neptune_offline
	parser.add_argument('--salve_image', action="store_true")				# true se passou --salve_image
	parser.add_argument('--show_image', action="store_true")				# true se passou --show_image
	parser.add_argument('--show_image_itk', action="store_true")			# true se passou --show_image_itk
	parser.add_argument('--debug', action="store_true") 					# true se passou --debug

	args = parser.parse_args()
	args.cli_args = ' '.join(cli_args)
	args.approachs_used_3d = ['vnet3d_monai','unet3d','unet3d_diedre','attentionUnet3D','residual_unet','munet3d']

	#parser.print_help()

	if (any(x==args.approach for x in args.approachs_used_3d)):
		print('Utilizando abordagem 3D.')
		args.plano = None
		args.spacing = None

	if (args.snout==6):
		args.labels_name = [1,2,3,4,5]
	elif (args.snout==7):
		args.labels_name = [7,8,4,5,6,512]
		args.sexperiment_name = "Lung lobe and bronchi segmentation on CT images"

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