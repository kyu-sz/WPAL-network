#!/usr/bin/env python

# --------------------------------------------------------------------
# This file is part of WMA Network.
# 
# WMA Network is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# WMA Network is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with WMA Network.  If not, see <http://www.gnu.org/licenses/>.
# --------------------------------------------------------------------

"""Compute minibatch blobs for training an AM network."""

import numpy as np
import numpy.random as npr
import cv2
from wma_net.config import config
from utils.blob import prep_img_for_blob, img_list_to_blob


def get_minibatch(img_paths, labels):
    """Construct a minibatch with given image paths and corresponding labels."""
    num_images = len(img_paths)

    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(config.TRAIN.SCALES),
                                    size=num_images)

    # Get the input image blob, formatted for caffe
    img_blob = _get_image_blob(img_paths, random_scale_inds)
    attr_blob = _get_attr_blob(labels)

    blobs = {'data': img_blob, 'attr': labels}

    return blobs


def _get_attr_blob(labels):
    """Builds an input blob from the labels"""
    blob = np.zeros((labels.shape[0], 1, 1, labels.shape[1]),
                    dtype=np.float32)


def _get_image_blob(img_paths, scale_inds):
    """Builds an input blob from the images at the specified
    scales.
    """
    num_images = len(img_paths)
    processed_ims = []
    img_scales = []
    for i in xrange(num_images):
        img = cv2.imread(img_paths[i])
        target_size = config.TRAIN.SCALES[scale_inds[i]]
        img, img_scale = prep_img_for_blob(img, config.PIXEL_MEANS, target_size,
                                        config.TRAIN.MAX_SIZE)
        img_scales.append(img_scale)
        processed_ims.append(img)

    # Create a blob to hold the input images
    blob = img_list_to_blob(processed_ims)

    return blob, img_scales
