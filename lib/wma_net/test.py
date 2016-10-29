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

"""Test a WPAL Network on an imdb (image database)."""

import cPickle
import os

import cv2
import numpy as np
from utils.blob import img_list_to_blob
from utils.timer import Timer

from config import config


def _get_image_blob(img):
    """Converts an image into a network input.
    Arguments:
        img (ndarray): a color image in BGR order
    Returns:
        blob (ndarray): a data blob holding the image
        img_scale_factor (double): image scale (relative to img) used
    """
    img_orig = img.astype(np.float32, copy=True)
    img_orig -= config.PIXEL_MEANS

    img_shape = img_orig.shape
    img_size_min = np.min(img_shape[0:2])
    img_size_max = np.max(img_shape[0:2])

    processed_images = []
    
    target_size = config.TEST.SCALE
    img_scale_factor = float(target_size) / float(img_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(img_scale_factor * img_size_max) > config.TEST.MAX_SIZE:
        img_scale_factor = float(config.TEST.MAX_SIZE) / float(img_size_max)
    img = cv2.resize(img_orig, None, None, fx=img_scale_factor, fy=img_scale_factor,
                     interpolation=cv2.INTER_LINEAR)
    processed_images.append(img)

    # Create a blob to hold the input images
    blob = img_list_to_blob(processed_images)

    return blob, img_scale_factor


def _get_blobs(im):
    """Convert an image into network inputs."""
    blobs = {'data' : None}
    blobs['data'], img_scale_factor = _get_image_blob(im)
    return blobs, img_scale_factor

def _attr_group_norm(pred, group):
    for i in group:
        pred[i] = 1 if pred[i] == max(pred[group]) else 0
    return pred

def recognize_attr(net, img, attr_group):
    """Recognize attributes in a pedestrian image.

    Arguments:
    	net (caffe.Net): WMA network to use.
    	img (ndarray): color image to test (in BGR order)
        attr_group(list of ranges): a list of ranges, each contains indexes
                                    of attributes that mutually exclude each other.

    Returns:
    	attributes (ndarray): K x 1 array of predicted attributes. (K is
    	    specified by database or the net)
    """
    blobs, img_scale_factor = _get_blobs(img)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))

    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    blobs_out = net.forward(**forward_kwargs)

    pred = np.average(blobs_out['pred'], axis=0)
    
    for group in attr_group:
        pred = _attr_group_norm(pred, group)
    
    for i in xrange(pred.shape[0]):
        pred[i] = 0 if pred[i] < 0.5 else 1

    return pred


def test_net(net, db, output_dir, vis=False):
    """Test a Weakly-supervised Pedestrian Attribute Localization Network on an image database."""

    num_images = len(db.test_ind)

    all_attrs = [[] for _ in xrange(num_images)]

    # timers
    _t = {'recognize_attr' : Timer()}
    
    cnt = 0
    for i in db.test_ind:
        img = cv2.imread(db.get_img_path(i))
        _t['recognize_attr'].tic()
        attr = recognize_attr(net, img, db.attr_group)
        _t['recognize_attr'].toc()
        all_attrs[cnt] = attr
        cnt += 1

        if cnt % 100 == 0:
            print 'recognize_attr: {:d}/{:d} {:.3f}s' \
                  .format(cnt, num_images, _t['recognize_attr'].average_time)

    attr_file = os.path.join(output_dir, 'attributes.pkl')
    with open(attr_file, 'wb') as f:
        cPickle.dump(all_attrs, f, cPickle.HIGHEST_PROTOCOL)

    print 'Mean accuracy:', db.evaluate_mA(all_attrs, db.test_ind)
    
    acc, prec, rec, f1 = db.evaluate_example_based(all_attrs, db.test_ind)

    print 'Acc={:f} Prec={:f} Rec={:f} F1={:f}'.format(acc, prec, rec, f1)
