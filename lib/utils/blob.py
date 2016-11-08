# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Blob helper functions."""

import math

import cv2
import numpy as np
from wpal_net.config import cfg


def img_list_to_blob(images):
    """Convert a list of images into a network input.
    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([img.shape for img in images]).max(axis=0)
    num_images = len(images)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in xrange(num_images):
        img = images[i]
        blob[i, 0:img.shape[0], 0:img.shape[1], :] = img
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob


def prep_img_for_blob(img, pixel_means, target_size, max_area, min_size):
    """Mean subtract and scale an image for use in a blob."""
    img = img.astype(np.float32, copy=False)
    img -= pixel_means
    img_shape = img.shape
    img_size_min = np.min(img_shape[0:2])
    img_size_max = np.max(img_shape[0:2])
    img_scale = float(target_size) / float(img_size_max)
    # Prevent the scaled area from being more than MAX_AREA
    if np.round(img_scale * img_size_min * img_scale * img_size_max) > max_area:
        img_scale = math.sqrt(float(max_area) / float(img_size_min * img_size_max))
    # Prevent the shorter sides from being less than MIN_SIZE
    if np.round(img_scale * img_size_min < min_size):
        img_scale = np.round(min_size / img_size_min) + 1
    img = cv2.resize(img, None, None, fx=img_scale, fy=img_scale, interpolation=cv2.INTER_LINEAR)
    img = cv2.warpAffine(img,
                         cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2),
                                                 np.random.randint(-15, 15), 1),
                         (img.shape[1], img.shape[0]))

    # Perform RGB Jittering
    h, w, c = img.shape
    zitter = np.zeros_like(img)
    for i in xrange(c):
        zitter[:, :, i] = np.random.randint(0, cfg.TRAIN.RGB_JIT, (h, w)) - cfg.TRAIN.RGB_JIT / 2
    img = cv2.add(img, zitter)

    return img, img_scale
