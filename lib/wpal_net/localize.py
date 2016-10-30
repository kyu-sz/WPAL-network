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

import cv2
import numpy as np

from config import cfg
from recog import recognize_attr


def learn_bounding(net, db, output_dir):
    bounding = np.ndarray((db.num_attr, cfg.NUM_DETECTOR))  # bounding between attribute and detector
    for i in db.train_ind:
        img = cv2.imread(db.get_img_path(i))
        attr, _, score = recognize_attr(net, img, db.attr_group)
        for j in xrange(db.labels[i].__len__()):
            if db.labels[i][j] > 0.5:
                bounding[j] += score
    bounding_file = os.path.join(output_dir, 'bounding.pkl')
    with open(bounding_file, 'wb') as f:
        cPickle.dump(bounding, f, cPickle.HIGHEST_PROTOCOL)

    detector_rank = [[j[0]
                      for j in sorted(enumerate(bounding[i]), key=lambda x: x[1])]
                     for i in xrange(bounding.__len__())]
    high_scores = [[j[1]
                    for j in sorted(enumerate(bounding[i]), key=lambda x: x[1])]
                   for i in xrange(bounding.__len__())]

    t = sum(high_scores.all()) / 32 / db.num_attr
    mat = np.ndarray((db.num_attr, cfg.NUM_DETECTOR))
    for i in xrange(detector_rank.size[0]):
        for j in xrange(32):
            mat[i][j] = t
    for i in xrange(cfg.NUM_DETECTOR):
        utilized = 0
        for j in xrange(detector_rank.size[0]):
            for k in xrange(32):
                if detector_rank[j][k] == i:
                    utilized = 1
                    break
            if utilized:
                break

    print detector_rank


def gaussian_filter(size, y, x, var=1):
    filter_map = np.ndarray(size)
    for i in xrange(0, size[0]):
        for j in xrange(0, size[1]):
            filter_map[i][j] = math.exp(-(math.pow(i - x, 2) + math.pow(j - y, 2)) / 2 / var) / 2 / var / math.pi
    return filter_map


if __name__ == '__main__':
    print gaussian_filter((8, 3), 2, 1)
