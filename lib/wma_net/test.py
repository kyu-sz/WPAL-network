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

"""Test a AM network on an imdb (image database)."""

import cPickle
import os

import cv2
import numpy as np
from utils.timer import Timer


def _get_image_blob(img):
    """Converts an image into a network input.
    Arguments:
        img (ndarray): a color image in BGR order
    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    img_orig = img.astype(np.float32, copy=True)
    img_orig -= config.PIXEL_MEANS

    img_shape = img_orig.shape
    img_size_min = np.min(img_shape[0:2])
    img_size_max = np.max(img_shape[0:2])

    processed_images = []
    img_scale_factors = []

    for target_size in config.TEST.SCALES:
        img_scale = float(target_size) / float(img_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(img_scale * img_size_max) > config.TEST.MAX_SIZE:
            img_scale = float(config.TEST.MAX_SIZE) / float(img_size_max)
        img = cv2.resize(img_orig, None, None, fx=img_scale, fy=img_scale,
                         interpolation=cv2.INTER_LINEAR)
        img_scale_factors.append(img_scale)
        processed_images.append(img)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_images)

    return blob, np.array(img_scale_factors)


def _get_blobs(im):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    return blobs, im_scale_factors


def recognize_attr(net, img):
    """Recognize attributes in a pedestrian image.

    Arguments:
    	net (caffe.Net): WMA network to use.
    	img (ndarray): color image to test (in BGR order)

    Returns:
    	attributes (ndarray): K x 1 array of predicted attributes. (K is
    	    specified by database or the net)
    """
    blobs, im_scale_factors = _get_blobs(img)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))

    raise NotImplementedError("Attribute recognition not implemented error")


def test_net(net, db, output_dir, vis=False):
    """Test a WMA Network on an image database."""

    num_images = len(db.test_ind)

    all_attrs = [[] for _ in xrange(db.num_attrs)]

    # timers
    _t = {'recognize_attr' : Timer()}

    cnt = 0
    for i in db.test_ind:
        im = cv2.imread(db.get_img_path(i))
        _t['recognize_attr'].tic()
        attr = recognize_attr(net, im)
        _t['recognize_attr'].toc()
        all_attrs[cnt] = attr
        ++cnt

        print 'recognize_attr: {:d}/{:d} {:.3f}s' \
              .format(cnt, num_images, _t['recognize_attr'].average_time)

    attr_file = os.path.join(output_dir, 'attributes.pkl')
    with open(attr_file, 'wb') as f:
        cPickle.dump(attr, f, cPickle.HIGHEST_PROTOCOL)

    print 'Evaluating attributes'
    db.evaluate_detections(attr, db.test_ind, output_dir)