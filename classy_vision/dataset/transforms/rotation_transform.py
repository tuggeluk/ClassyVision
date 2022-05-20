#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from . import register_transform
from .classy_transform import ClassyTransform
from scipy import ndimage
import PIL



@register_transform("RotationTransform")
class RotationTransform(ClassyTransform):
    """
    Rotates an image
    """

    def __call__(self, img):
        angle = int(torch.randint(low=0, high=259, size=(1, 1)))
        img = ndimage.rotate(img, angle, reshape=False)
        return img
