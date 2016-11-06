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
import os

import cv2
import numpy as np

from config import cfg
from recog import recognize_attr


def estimate_param(net, db, output_dir, res_file):
    attrs = []
    scores = []
    labels = []

    if res_file == None:
        cnt = 0
        for i in db.train_ind:
            img = cv2.imread(db.get_img_path(i))
            attr, _, _, _, score = recognize_attr(net, img, db.attr_group)
            attrs.append(attr)
            scores.append([score[x][0][0] for x in range(len(score))])
            labels.append(db.labels[i])
            cnt += 1
            if cnt % 1000 == 0:
                print 'Tested: {}/{}'.format(cnt, db.train_ind.__len__())

        val_file = os.path.join(output_dir, 'val.pkl')
        with open(val_file, 'wb') as f:
            cPickle.dump({'attrs': attrs, 'scores': scores}, f, cPickle.HIGHEST_PROTOCOL)
    else:
        print 'Loading stored results.'
        f = open(res_file, 'rb')
        pack = cPickle.load(f)
        attrs = pack['attrs']
        scores = pack['scores']
        labels = db.labels[db.train_ind]
        print 'Stored results loaded!'

    # Calculate average score of a detector
    ave = np.zeros(cfg.NUM_DETECTOR)
    for v in scores:
        ave += np.array(v)
    ave /= len(scores)

    binding = np.zeros((db.num_attr, cfg.NUM_DETECTOR))  # binding between attribute and detector
    # Estimate detector binding
    for i in xrange(len(attrs[0])):
        pos_ind = np.where(labels[:][i] > 0.5)[0]
        for j in pos_ind:
            binding[i] += np.array(scores[j]) / (ave * len(pos_ind))

    detector_file = os.path.join(output_dir, 'detector.pkl')
    with open(detector_file, 'wb') as f:
        cPickle.dump({'ave':ave,'binding':binding}, f, cPickle.HIGHEST_PROTOCOL)

