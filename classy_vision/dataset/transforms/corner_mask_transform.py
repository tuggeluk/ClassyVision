#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

import numpy as np
from . import register_transform
import PIL
from .classy_transform import ClassyTransform


@register_transform("CornerMask")
class CornerMask(ClassyTransform):
    """
    Mask out corners of image such that a circle of unmasked pixels remains
    """

    def __call__(self, img):
        img = np.array(img)
        x = np.arange(0, img.shape[0], 1) - np.floor(img.shape[0] / 2)
        y = np.arange(0, img.shape[1], 1) - np.floor(img.shape[1] / 2)
        xx, yy = np.meshgrid(x, y)
        mask = (np.sqrt((xx * xx) + (yy * yy)) - img.shape[0] / 2) > 0
        img[mask] = 0
        #PIL.Image.fromarray(img).show()
        return img
