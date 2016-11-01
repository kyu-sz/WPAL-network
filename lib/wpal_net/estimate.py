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

import cPickle
import math
import os
from random import randint

import cv2
import numpy as np
from utils.evaluate import mA

from config import cfg
from recog import recognize_attr


def estimate_param(net, db, output_dir, res_file):
    binding = np.ndarray((db.num_attr, cfg.NUM_DETECTOR))  # binding between attribute and detector
    attrs = []
    scores = []
    labels = []

    if res_file == None:
        for i in db.train_ind:
            img = cv2.imread(db.get_img_path(i))
            attr, _, score = recognize_attr(net, img, db.attr_group)
            attrs.append(attr)
            scores.append(score)
            labels.append(db.labels[i])
        cPickle.dump({'attrs': attrs, 'scores': scores})
    else:
        f = open(res_file, 'rb')
        pack = cPickle.load(f)
        attrs = pack['attrs']
        scores = pack['scores']

    # Find challenging attributes
    _, challenging = mA(attrs, db.labels[db.train_ind])

    # Estimate detector binding
    for i in xrange(attrs.__len__()):
        for j in xrange(labels[i].__len__()):
            if db.labels[i][j] > 0.5:
                binding[j] += scores[i]
    binding_file = os.path.join(output_dir, 'binding.pkl')
    with open(binding_file, 'wb') as f:
        cPickle.dump(binding, f, cPickle.HIGHEST_PROTOCOL)

    # Sort the detectors by scores.
    detector_rank = [[j[0]
                      for j in sorted(enumerate(binding[i]), key=lambda x: x[1])]
                     for i in xrange(binding.__len__())]
    high_scores = [[j[1]
                    for j in sorted(enumerate(binding[i]), key=lambda x: x[1])]
                   for i in xrange(binding.__len__())]

    # Estimate supervision threshold
    t = sum(high_scores.all()) / 32 / db.num_attr
    mat = np.ndarray((db.num_attr, cfg.NUM_DETECTOR))
    # Find binding of the first 32 detectors with highest scores.
    for i in xrange(detector_rank.size[0]):
        for j in xrange(32):
            mat[i][j] = 1
    # Assign the rest detectors randomly to challenging attributes.
    for i in xrange(cfg.NUM_DETECTOR):
        utilized = 0
        for j in xrange(detector_rank.size[0]):
            for k in xrange(32):
                if detector_rank[j][k] == i:
                    utilized = 1
                    break
            if utilized:
                break
        if not utilized:
            mat[challenging[randint(0, challenging.__len__() - 1)]][i] = 1

    dtl_file = os.path.join(output_dir, 'detector_threshold.pkl')
    with open(dtl_file, 'wb') as f:
        cPickle.dump({
            't':   t,
            'mat': mat
        }, f, cPickle.HIGHEST_PROTOCOL)


def gaussian_filter(size, y, x, var=1):
    filter_map = np.ndarray(size)
    for i in xrange(0, size[0]):
        for j in xrange(0, size[1]):
            filter_map[i][j] = math.exp(-(math.pow(i - x, 2) + math.pow(j - y, 2)) / 2 / var) / 2 / var / math.pi
    return filter_map


if __name__ == '__main__':
    print gaussian_filter((8, 3), 2, 1)
