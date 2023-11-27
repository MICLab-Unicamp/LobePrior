#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def printImage3D(image, label):
	print(image.shape, label.shape)
	if (len(image.shape)==4):
		image = image[0,:,:,:]
		label = label[0,:,:,:]
	elif (len(image.shape)==5):
		image = image[0,0,:,:,:]
		label = label[0,0,:,:,:]
	print(image.shape, label.shape)

	f, (plot1, plot2, plot3, plot4, plot5, plot6) = plt.subplots(1, 6, figsize = (12, 6))
	plot1.set_title(f"Axial")
	plot1.set_axis_off()
	plot1.imshow(image[image.shape[0]//2,:,:])
	plot2.imshow(label[label.shape[0]//2,:,:])
	plot2.set_title(f"Axial")
	plot2.set_axis_off()
	plot3.set_title(f"Sagital")
	plot3.set_axis_off()
	plot3.imshow(image[:,image.shape[1]//2,:])
	plot4.imshow(label[:,label.shape[1]//2,:])
	plot4.set_title(f"Sagital")
	plot4.set_axis_off()
	plot5.set_title(f"Coronal")
	plot5.set_axis_off()
	plot5.imshow(image[:,:,image.shape[2]//2])
	plot6.imshow(label[:,:,label.shape[2]//2])
	plot6.set_title(f"Coronal")
	plot6.set_axis_off()

	plt.show()
	plt.close()

def showImagesToOneHot3D(image, label, n_slice=None, message='Check'):
	if n_slice is None:
		n_slice = image.shape[1]//2

	print('Shape',image.shape, label.shape)

	plt.figure(message, (12, 6))
	plt.subplot(171)
	plt.title("Image")
	plt.axis('off')
	plt.imshow(image[0,n_slice, :, :])

	plt.subplot(172)
	plt.title("Background")
	plt.axis('off')
	plt.imshow(label[0,n_slice, :, :])
	plt.subplot(173)
	plt.title("RUL")
	plt.axis('off')
	plt.imshow(label[1,n_slice, :, :])
	plt.subplot(174)
	plt.title("RML")
	plt.axis('off')
	plt.imshow(label[2,n_slice, :, :])
	plt.subplot(175)
	plt.title("RLL")
	plt.axis('off')
	plt.imshow(label[3,n_slice, :, :])
	plt.subplot(176)
	plt.title("LUL")
	plt.axis('off')
	plt.imshow(label[4,n_slice, :, :])
	plt.subplot(177)
	plt.title("LLL")
	plt.axis('off')
	plt.imshow(label[5,n_slice, :, :])

	plt.tight_layout()
	plt.show()
	plt.close()

def showImagesToOneHot3DOverlay(image, label, n_slice=None, message='Check'):
	if n_slice==None:
		n_slice = image.shape[1]//2

	print('Shape (showImagesToOneHot3DOverlay):',image.shape, label.shape)

	plt.figure(message, (12, 6))
	plt.subplot(171)
	plt.title("Image")
	plt.axis('off')
	plt.imshow(image[0,n_slice, :, :], cmap="gray")

	plt.subplot(172)
	plt.title("Background")
	plt.axis('off')
	plt.imshow((image[0,n_slice,:,:].squeeze())/2, cmap="gray")
	plt.imshow((np.array(label.argmax(axis=0)).astype(np.uint8)[n_slice,:,:].squeeze())/2, alpha=0.6)
	plt.subplot(173)
	plt.title("LUL")
	plt.axis('off')
	plt.imshow((image[0,n_slice,:,:].squeeze())/2, cmap="gray")
	plt.imshow((label[1,n_slice,:,:].squeeze())/2, cmap="Reds", alpha=0.6)
	plt.subplot(174)
	plt.title("LLL")
	plt.axis('off')
	plt.imshow((image[0,n_slice,:,:].squeeze())/2, cmap="gray")
	plt.imshow((label[2,n_slice,:,:].squeeze())/2, cmap="Greens", alpha=0.6)
	plt.subplot(175)
	plt.title("RUL")
	plt.axis('off')
	plt.imshow((image[0,n_slice,:,:].squeeze())/2, cmap="gray")
	plt.imshow((label[3,n_slice,:,:].squeeze())/2, cmap="Blues", alpha=0.6)
	plt.subplot(176)
	plt.title("RML")
	plt.axis('off')
	plt.imshow((image[0,n_slice,:,:].squeeze())/2, cmap="gray")
	plt.imshow((label[4,n_slice,:,:].squeeze())/2, cmap="Oranges", alpha=0.6)
	plt.subplot(177)
	plt.title("RLL")
	plt.axis('off')
	plt.imshow((image[0,n_slice,:,:].squeeze())/2, cmap="gray")
	plt.imshow((label[5,n_slice,:,:].squeeze())/2, cmap="Purples", alpha=0.6)

	plt.tight_layout()
	plt.show()
	plt.close()

def showImagesToOneHot3DTemplateOverlay(image, label, n_slice=None, message='Check'):
	if n_slice==None:
		n_slice = image.shape[1]//2

	print('Shape (showImagesToOneHot3DTemplateOverlay):',image.shape, label.shape)

	plt.figure(message, (12, 6))
	plt.subplot(171)
	plt.title("Image")
	plt.axis('off')
	plt.imshow(image[0,n_slice, :, :], cmap="gray")

	plt.subplot(172)
	plt.title("Background")
	plt.axis('off')
	plt.imshow((image[0,n_slice,:,:].squeeze())/2, cmap="gray")
	plt.imshow((np.array(label.argmax(axis=0)).astype(np.uint8)[n_slice,:,:].squeeze())/2, alpha=0.6)
	plt.subplot(173)
	plt.title("LUL")
	plt.axis('off')
	plt.imshow((image[0,n_slice,:,:].squeeze())/2, cmap="gray")
	plt.imshow((label[0,n_slice,:,:].squeeze())/2, cmap="Reds", alpha=0.6)
	plt.subplot(174)
	plt.title("LLL")
	plt.axis('off')
	plt.imshow((image[0,n_slice,:,:].squeeze())/2, cmap="gray")
	plt.imshow((label[1,n_slice,:,:].squeeze())/2, cmap="Greens", alpha=0.6)
	plt.subplot(175)
	plt.title("RUL")
	plt.axis('off')
	plt.imshow((image[0,n_slice,:,:].squeeze())/2, cmap="gray")
	plt.imshow((label[2,n_slice,:,:].squeeze())/2, cmap="Blues", alpha=0.6)
	plt.subplot(176)
	plt.title("RML")
	plt.axis('off')
	plt.imshow((image[0,n_slice,:,:].squeeze())/2, cmap="gray")
	plt.imshow((label[3,n_slice,:,:].squeeze())/2, cmap="Oranges", alpha=0.6)
	plt.subplot(177)
	plt.title("RLL")
	plt.axis('off')
	plt.imshow((image[0,n_slice,:,:].squeeze())/2, cmap="gray")
	plt.imshow((label[4,n_slice,:,:].squeeze())/2, cmap="Purples", alpha=0.6)

	plt.tight_layout()
	plt.show()
	plt.close()