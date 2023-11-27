#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import cc3d
import numpy as np
from operator import itemgetter

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


def post_processing(output, verbose=None):
    '''
    Post processing pipeline for lung and covid segmentation
    Input should be numpy activations
    '''
    assert output.ndim == 4, "Input to lung and covid post processing has to be four dimensional"

    if verbose is not None:
        verbose.write("Unpacking outputs...")
    lung, covid = (output[0] > 0.5).astype(np.int32), (output[1] > 0.5).astype(np.int32)
    if verbose is not None:
        verbose.write("Calculating lung connected components...")
    lung, lung_lc, lung_labeled = get_connected_components(lung, return_largest=2)

    if verbose is not None:
        verbose.write("Extracting first and second largest components...")
    first_component = lung_labeled == lung_lc[0][0]
    try:
        second_component = lung_labeled == lung_lc[1][0]
    except IndexError:
        verbose.write("WARNING: Was not able to get a second component.")
        second_component = np.zeros_like(first_component)

    if verbose is not None:
        verbose.write("WARNING: Skipping lung split.")
    lung = first_component + second_component
    covid = covid*lung

    return lung.astype(np.uint8), covid.astype(np.uint8)


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

import fill_voids
import skimage.morphology
import scipy.ndimage as ndimage
from tqdm import tqdm

def bbox_3D(labelmap, margin=2):
    shape = labelmap.shape
    r = np.any(labelmap, axis=(1, 2))
    c = np.any(labelmap, axis=(0, 2))
    z = np.any(labelmap, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    rmin -= margin if rmin >= margin else rmin
    rmax += margin if rmax <= shape[0] - margin else rmax
    cmin, cmax = np.where(c)[0][[0, -1]]
    cmin -= margin if cmin >= margin else cmin
    cmax += margin if cmax <= shape[1] - margin else cmax
    zmin, zmax = np.where(z)[0][[0, -1]]
    zmin -= margin if zmin >= margin else zmin
    zmax += margin if zmax <= shape[2] - margin else zmax

    if rmax-rmin == 0:
        rmax = rmin+1

    return np.asarray([rmin, rmax, cmin, cmax, zmin, zmax])

def keep_largest_connected_component(mask):
    mask = skimage.measure.label(mask)
    regions = skimage.measure.regionprops(mask)
    resizes = np.asarray([x.area for x in regions])
    max_region = np.argsort(resizes)[-1] + 1
    mask = mask == max_region
    return mask

def postprocessing(label_image, spare=[]):
    '''some post-processing mapping small label patches to the neighbout whith which they share the
        largest border. All connected components smaller than min_area will be removed
    '''

    label_image = (label_image > 0.5)

    # merge small components to neighbours
    regionmask = skimage.measure.label(label_image)
    origlabels = np.unique(label_image)
    origlabels_maxsub = np.zeros((max(origlabels) + 1,), dtype=np.uint32)  # will hold the largest component for a label
    regions = skimage.measure.regionprops(regionmask, label_image)
    regions.sort(key=lambda x: x.area)
    regionlabels = [x.label for x in regions]

    # will hold mapping from regionlabels to original labels
    region_to_lobemap = np.zeros((len(regionlabels) + 1,), dtype=np.uint8)
    for r in regions:
        r_max_intensity = int(r.max_intensity)
        if r.area > origlabels_maxsub[r_max_intensity]:
            origlabels_maxsub[r_max_intensity] = r.area
            region_to_lobemap[r.label] = r_max_intensity

    for r in tqdm(regions):
        r_max_intensity = int(r.max_intensity)
        if (r.area < origlabels_maxsub[r_max_intensity] or r_max_intensity in spare) and r.area>2: # area>2 improves runtime because small areas 1 and 2 voxel will be ignored
            bb = bbox_3D(regionmask == r.label)
            sub = regionmask[bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]]
            dil = ndimage.binary_dilation(sub == r.label)
            neighbours, counts = np.unique(sub[dil], return_counts=True)
            mapto = r.label
            maxmap = 0
            myarea = 0
            for ix, n in enumerate(neighbours):
                if n != 0 and n != r.label and counts[ix] > maxmap and n not in spare:
                    maxmap = counts[ix]
                    mapto = n
                    myarea = r.area
            regionmask[regionmask == r.label] = mapto
            # print(str(region_to_lobemap[r.label]) + ' -> ' + str(region_to_lobemap[mapto])) # for debugging
            if regions[regionlabels.index(mapto)].area == origlabels_maxsub[
                int(regions[regionlabels.index(mapto)].max_intensity)]:
                origlabels_maxsub[int(regions[regionlabels.index(mapto)].max_intensity)] += myarea
            regions[regionlabels.index(mapto)].__dict__['_cache']['area'] += myarea

    outmask_mapped = region_to_lobemap[regionmask]
    outmask_mapped[np.isin(outmask_mapped, spare)] = 0

    if outmask_mapped.shape[0] == 1:
        # holefiller = lambda x: ndimage.morphology.binary_fill_holes(x[0])[None, :, :] # This is bad for slices that show the liver
        holefiller = lambda x: skimage.morphology.area_closing(x[0].astype(int), area_threshold=64)[None, :, :] == 1
    else:
        holefiller = fill_voids.fill

    outmask = np.zeros(outmask_mapped.shape, dtype=np.uint8)
    for i in np.unique(outmask_mapped)[1:]:
        outmask[holefiller(keep_largest_connected_component(outmask_mapped == i))] = i

    return outmask

def pos_processed(image):
    assert len(image.shape)==5

    print('Start of post processing')
    if torch.is_tensor(image):
        image = image.numpy()
    for channel in range(1, image.shape[1]):
        #print(f'channel {channel}: {image[channel,...].shape}')
        image[0, channel] = postprocessing(image[0, channel])
        #image[0, channel] = post_processing(image[0, channel])
    #image = torch.from_numpy(image)
    image = torch.tensor(image)
    print('End of post processing')

    return image