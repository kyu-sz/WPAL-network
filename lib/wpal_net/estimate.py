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


def estimate_param(net, db, output_dir, res_file, save_res=False):
    attrs = []
    scores = []
    labels = []

    if res_file == None:
        cnt = 0
        for i in db.train_ind:
            img = cv2.imread(db.get_img_path(i))
            attr, _, _, _, score = recognize_attr(net, img, db.attr_group)
            attrs.append(attr)
            scores.append([x for x in score])
            labels.append(db.labels[i])
            cnt += 1
            if cnt % 1000 == 0:
                print 'Tested: {}/{}'.format(cnt, db.train_ind.__len__())

        if save_res:
            print 'Saving results...'
            val_file = os.path.join(output_dir, 'val.pkl')
            with open(val_file, 'wb') as f:
                cPickle.dump({'attrs': attrs, 'scores': scores}, f, cPickle.HIGHEST_PROTOCOL)
            print 'Results stored to {}!'.format(val_file)
    else:
        print 'Loading stored results from {}.'.format(res_file)
        pack = cPickle.load(open(res_file, 'rb'))
        attrs = pack['attrs']
        scores = pack['scores']
        labels = db.labels[db.train_ind]
        print 'Stored results loaded!'

    pos_ave = np.zeros((db.num_attr, len(scores[0])))  # binding between attribute and detector or detector bin
    neg_ave = np.zeros((db.num_attr, len(scores[0])))  # binding between attribute and detector or detector bin
    # Estimate detector binding
    for i in xrange(db.num_attr):
        pos_ind = np.where(np.array([x[i] for x in labels]) > 0.5)[0]
        neg_ind = np.where(np.array([x[i] for x in labels]) < 0.5)[0]
        print 'For attr {}: pos={}; neg={}'.format(i, len(pos_ind),len(neg_ind))
        pos_sum = np.zeros(len(scores[0]), dtype=float)
        neg_sum = np.zeros(len(scores[0]), dtype=float)
        for j in pos_ind:
            pos_sum += np.array(scores[j])
        for j in neg_ind:
            neg_sum += np.array(scores[j])
        pos_ave[i] = pos_sum / len(pos_ind)
        neg_ave[i] = neg_sum / len(neg_sum)
        print 'Estimated attr {}/{}'.format(i, db.num_attr)
    binding = np.exp(pos_ave / neg_ave)

    detector_file = os.path.join(output_dir, 'detector.pkl')
    with open(detector_file, 'wb') as f:
        cPickle.dump({'pos_ave':pos_ave,'neg_ave':neg_ave,'binding':binding}, f, cPickle.HIGHEST_PROTOCOL)

