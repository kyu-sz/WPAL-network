#!/usr/bin/env python

# --------------------------------------------------------------------
# This file is part of
# Weakly-supervised Pedestrian Attribute Localization Network.
#
# Weakly-supervised Pedestrian Attribute Localization Network
# is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Weakly-supervised Pedestrian Attribute Localization Network
# is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Weakly-supervised Pedestrian Attribute Localization Network.
# If not, see <http://www.gnu.org/licenses/>.
# --------------------------------------------------------------------

"""Compute minibatch blobs for training an WPAL Network."""

import cv2
import numpy as np
import numpy.random as npr
from utils.blob import img_list_to_blob, prep_img_for_blob
from wpal_net.config import config


def get_minibatch(img_paths, labels, flip, flip_attr_pairs, weight):
    """Construct a minibatch with given image paths and corresponding labels."""
    num_images = len(img_paths)

    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(config.TRAIN.SCALES),
                                    size=num_images)

    # Get the input image blob, formatted for caffe
    img_blob, img_scales = _get_image_blob(img_paths, random_scale_inds, flip)
    attr_blob = _get_attr_blob(labels, flip, flip_attr_pairs)
    weight_blob = _get_weight_blob(labels, weight)

    blobs = {'data': img_blob, 'attr': attr_blob, 'weight': weight_blob}

    return blobs


def _flip_labels(labels, flip, flip_attr_pairs):
    """Horizontally flip the labels according to flipping flags.
    labels: 1-dimensional numpy array.
    flip:   corresponding flipping flag array.
    flip_attr_pairs: A list of attribute pairs to be flipped.
    """
    for pair in flip_attr_pairs:
        face_left_ind = [pair[0]]
        face_right_ind = [pair[1]]
        temp = labels[face_right_ind]
        labels[face_right_ind] = labels[face_left_ind]
        labels[face_left_ind] = temp
    return labels


def _get_weight_blob(labels, weight):
    """Builds an input blob from the labels"""
    blob = np.zeros((labels.__len__(), 1, 1, labels[0].__len__()),
                    dtype=np.float32)
    for i in xrange(labels.__len__()):
        blob[i, :, :, :] = weight

    return blob


def _get_attr_blob(labels, flip, flip_attr_pairs):
    """Builds an input blob from the labels"""
    blob = np.zeros((labels.__len__(), 1, 1, labels[0].__len__()),
                    dtype=np.float32)
    for i in xrange(labels.__len__()):
        blob[i, :, :, :] = _flip_labels(labels[i], flip[i], flip_attr_pairs)

    return blob


def _get_image_blob(img_paths, scale_inds, flip):
    """Builds an input blob from the images at the specified
    scales.
    """
    num_images = len(img_paths)
    processed_imgs = []
    img_scales = []
    for i in xrange(num_images):
        img = cv2.imread(img_paths[i])
	"""Flip the image if required."""
        if flip[i]:
            img = cv2.flip(img, 1)
        target_size = config.TRAIN.SCALES[scale_inds[i]]
        img, img_scale = prep_img_for_blob(img, config.PIXEL_MEANS, target_size,
                                           config.TRAIN.MAX_AREA)
        img_scales.append(img_scale)
        processed_imgs.append(img)

    # Create a blob to hold the input images
    blob = img_list_to_blob(processed_imgs)

    return blob, img_scales
