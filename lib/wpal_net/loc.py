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

"""Test localization of a WPAL Network."""

import cPickle
import math
import os

import cv2
import numpy as np
from utils.timer import Timer

from estimate import gaussian_filter as gf
from recog import recognize_attr


def localize(net, db, output_dir, dweight, attr_id, vis=False, save_dir=None):
    """Test localization of a WPAL Network."""

    num_images = len(db.test_ind)

    all_attrs = [[] for _ in xrange(num_images)]

    # timers
    _t = {'recognize_attr' : Timer()}

    threshold = np.ones(db.num_attr) * 0.5;

    if attr_id == -1:
        # localize whole body outline
        attr_list = xrange(db.num_attr)
    else:
        # localize only one attribute
        attr_list = []
        attr_list.append(attr_id)

    cnt = 0
    for i in db.test_ind:
        img_path = db.get_img_path(i)
        img = cv2.imread(img_path)
        _t['recognize_attr'].tic()
        attr, heat3, heat4, heat5, score = recognize_attr(net, img, db.attr_group, threshold)
        _t['recognize_attr'].toc()
        all_attrs[cnt] = attr
        cnt += 1

        canvas = np.array(img, dtype=float)
        for attr in attr_list:
            w_func = lambda x: 0 if score[x] <= 0 else math.exp(score[x] + dweight[attr][x])
            w_sum = sum([w_func(j) for j in xrange(len(score))])
            get_heat_map = lambda x: heat3[x] if x < heat3.shape[0] else \
                                     heat4[x - heat3.shape[0]] if  x - heat3.shape[0] < heat4.shape[0] else \
                                     heat5[x - heat3.shape[0] - heat4.shape[0]]
            find_target = lambda x: np.where(np.array(get_heat_map(x)) == score[x])
            target = [find_target(j)[:][0] for j in xrange(len(score))]
            print target
            # Center of the feature.
            x = img.shape[1] * sum([w_func(j) / w_sum * target[j][1] / np.array(get_heat_map(j)).shape[1]
                                    for j in xrange(len(score))])
            y = img.shape[0] * sum([w_func(j) / w_sum * target[j][0] / np.array(get_heat_map(j)).shape[0]
                                    for j in xrange(len(score))]) 
            # Superposition of the heat maps.
            print y, x
            superposition = sum([w_func(j) / w_sum * img
                                 * gf(img.shape, img.shape[0] * target[j][0] / get_heat_map(j).shape[0],
                                                 img.shape[1] * target[j][1] / get_heat_map(j).shape[1])
                                 for j in xrange(len(score))])
            mean = (superposition.max() + superposition.min()) / 2
            range = superposition.max() - superposition.min()
            superposition = (superposition - mean) * 128 / range
            for j in xrange(img.shape[0]):
                for k in xrange(img.shape[1]):
                    if superposition[j][k] >= 0:
                        canvas[j][k][2] += superposition[j][k]
                    else:
                        canvas[j][k][0] -= superposition[j][k]
            print 'Attribute', attr, 'computed!'
        
        canvas = canvas.astype(int)
        for j in xrange(img.size[0]):
            for k in xrange(img.size[1]):
                canvas[j][k][0] = min(255, max(0, canvas[j][k][0]))
                canvas[j][k][2] = min(255, max(0, canvas[j][k][2]))

        if vis:
            cv2.imshow(db.get_img_path(i), canvas)
            cv2.waitKey(0)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, os.path.split(img_path)[1], img))

        if cnt % 100 == 0:
            print 'recognize_attr: {:d}/{:d} {:.3f}s' \
                  .format(cnt, num_images, _t['recognize_attr'].average_time)

    attr_file = os.path.join(output_dir, 'attributes.pkl')
    with open(attr_file, 'wb') as f:
        cPickle.dump(all_attrs, f, cPickle.HIGHEST_PROTOCOL)

    mA, challenging = db.evaluate_mA(all_attrs, db.test_ind)
    print 'mA={:f}'.format(mA)
    print 'Challenging attributes:', challenging
    
    acc, prec, rec, f1 = db.evaluate_example_based(all_attrs, db.test_ind)

    print 'Acc={:f} Prec={:f} Rec={:f} F1={:f}'.format(acc, prec, rec, f1)
