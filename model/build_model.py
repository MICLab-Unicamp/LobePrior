#!/usr/bin/env python
# -*- coding: utf-8 -*-

import monai

from .unet3d_pytorch import UNet3d
from .unet_diedre import UNet_Diedre
from .m3DUNet import Modified3DUNet
from .residual_unet.residual_unet import ResidualUNet3D
from .unet_grid_attention_3D import unet_grid_attention_3D
from .BB_unet3d_pytorch import BB_Unet_pytorch
from .BB_Unet_3D import BB_Unet

def build_model(hparams):
	if (hparams.approach == "unet3d"):
		model = UNet3d(in_channels = hparams.nin, n_classes = hparams.snout, s_channels = 32)
	elif (hparams.approach == "vnet3d_monai"):
		model = monai.networks.nets.VNet(in_channels = hparams.nin, out_channels = hparams.snout)
	elif (hparams.approach == "unet3d_diedre"):
		model = UNet_Diedre(n_channels=hparams.nin, n_classes=hparams.snout, norm=True, dim='3d', init_channel=16, joany_conv=False, dict_return=False)
	elif (hparams.approach == "munet3d"):
		model = Modified3DUNet(in_channels = hparams.nin, n_classes = hparams.snout)
	elif (hparams.approach == "residual_unet"):
		model = ResidualUNet3D(in_channels=hparams.nin,out_channels=hparams.snout)
	elif (hparams.approach == "attentionUnet3D"): 
		model = unet_grid_attention_3D(in_channels=hparams.nin, n_classes=hparams.snout)
	elif (hparams.approach == "UNETR"):
		model = monai.networks.nets.unetr(in_channels=hparams.nin,out_channels=hparams.snout,img_size=(64,256,256),feature_size=16,hidden_size=768,mlp_dim=3072,num_heads=12,pos_embed="perceptron",norm_name="instance",res_block=True,dropout_rate=0.0,)
	elif (hparams.approach == "SwinUNETR"):
		model = monai.networks.nets.swin_unetr(img_size=(128, 128, 128),in_channels=2,out_channels=hparams.snout,feature_size=48,drop_rate=0.0,attn_drop_rate=0.0,dropout_path_rate=0.0,use_checkpoint=True)
	elif (hparams.approach == "BB_Unet"):
		model = BB_Unet(BB_boxes=6)
	elif (hparams.approach == "BB_Unet_pytorch"):
		model = BB_Unet_pytorch(in_channels=2, n_classes=6, s_channels=64, BB_boxes=6)
	else:
		print("\tModelo não especificado em lightning.")

	return model
