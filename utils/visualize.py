#!/usr/bin/env python
# -*- coding: utf-8 -*-

import uuid
import subprocess
import numpy as np
import SimpleITK as sitk

def surface_render_itksnap(img: np.ndarray, int_tgt: np.ndarray = None, label='', block=False):
    '''
    Uses itksnap to render a numpy array and its tgt if given
    '''
    uid = uuid.uuid4()
    img_path = f"/tmp/{label}_itksnap_{uid}.nii.gz"
    if int_tgt is not None:
        tgt_path = f"/tmp/{label}_tgt_itksnap_{uid}.nii.gz"
        sitk_tgt = sitk.GetImageFromArray(int_tgt)
        sitk.WriteImage(sitk_tgt, tgt_path)

    sitk_image = sitk.GetImageFromArray(img)
    sitk.WriteImage(sitk_image, img_path)

    if block:
        if int_tgt is None:
            subprocess.run(["itksnap", "-g", img_path])
        else:
            subprocess.run(["itksnap", "-g", img_path, "-s", tgt_path])
    else:
        if int_tgt is None:
            subprocess.Popen(["itksnap", "-g", img_path])
        else:
            subprocess.Popen(["itksnap", "-g", img_path, "-s", tgt_path])