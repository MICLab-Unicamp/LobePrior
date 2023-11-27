#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
from typing import Tuple

"""## Transformadas (Data Augmentation ou Aumentação de Dados)
Transformadas são implementadas como classes chamadas pelo Dataset

Note que operações de Data Augmentation acontecem em tempo real durante o treino.
"""

class CTHUClip():
    '''
    Clip and normalize to [0-1] range, taking into account a constant intended maximum and minimum value 
    regardless of what is present in the image
    '''
    def __init__(self, vmin=-1024, vmax=600):
        self.vmin = vmin
        self.vmax = vmax

    def __call__(self, x_y: Tuple[np.ndarray, np.ndarray]):
        x, y = x_y
        if torch.is_tensor(x):
            x = torch.clip(x, self.vmin, self.vmax)
        elif isinstance(x, np.ndarray):
            x = np.clip(x, self.vmin, self.vmax)
        else:
            raise ValueError(f"Unsupported x type for CTHUClip {type(x)}")

        x = (x - self.vmin)/(self.vmax - self.vmin)

        if y is not None:
            return x, y
        else:
            return x

    def __str__(self):
        return f"CTHUClip vmin: {self.vmin} vmax: {self.vmax}"

class Clip():
	def __init__(self, min, max):
		self.min = min
		self.max = max

	def __call__(self, x_y: Tuple[np.ndarray, np.ndarray]):
		x, y = x_y
		x = np.clip(x, self.min, self.max)
		return x, y

class MinMaxNormalize():
	def __init__(self, vmin=-1024, vmax=600):
		self.vmin = vmin
		self.vmax = vmax

	def __call__(self, x_y: Tuple[np.ndarray, np.ndarray]):
		x, y = x_y
		x = (x - x.min()) / (x.max() - x.min())
		#x = (x - self.vmin)/(self.vmax - self.vmin)
		return x, y

class Standardization():
	def __call__(self, x_y: Tuple[np.ndarray, np.ndarray]):
		x, y = x_y
		x = (x - np.average(x)) / (np.std(x))
		return x, y
