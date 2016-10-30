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
import math
import os

import cv2
import numpy as np
from utils.timer import Timer

from localize import gaussian_filter as gf
from recog import recognize_attr


def test_net(net, db, output_dir, vis=False, detector_weight=None, save_file=None):
    """Test a Weakly-supervised Pedestrian Attribute Localization Network on an image database."""

    num_images = len(db.test_ind)

    all_attrs = [[] for _ in xrange(num_images)]

    # timers
    _t = {'recognize_attr' : Timer()}
    
    cnt = 0
    for i in db.test_ind:
        img = cv2.imread(db.get_img_path(i))
        _t['recognize_attr'].tic()
        attr, heat, score = recognize_attr(net, img, db.attr_group)
        _t['recognize_attr'].toc()
        all_attrs[cnt] = attr
        cnt += 1

        if vis:
            if detector_weight is None:
                print "Visualization need detector_weight to be not none!"
                vis = False
            else:
                w_func = lambda x: 0 if score[x] <= 0 else math.exp(score[x] + detector_weight[x])
                find_target = lambda x: np.where(heat[x] == score[x])
                w_sum = sum([w_func(j) for j in xrange(score.__len__())])
                x = img.shape[1] * sum([w_func[j] / w_sum * find_target(j)[1] / heat[j].shape[1]
                                        for j in xrange(score.__len__())])
                y = img.shape[0] * sum([w_func[j] / w_sum * find_target(j)[0] / heat[j].shape[0]
                                        for j in xrange(score.__len__())])
                superposition = sum([
                                        cv2.resize(heat[j] * gf(heat[j].size, y, x), img.shape)
                                        for j in xrange(score.__len__())
                                        ])
                mean = (superposition.max() + superposition.min()) / 2
                range = superposition.max() - superposition.min()
                superposition = (superposition - mean) * 128 / range
                canvas = np.ndarray(img)
                for j in xrange(img.size[0]):
                    for k in xrange(img.size[1]):
                        if superposition[j][k] >= 0:
                            canvas[j][k][2] = min(255, canvas[j][k][2] + superposition[j][k])
                        else:
                            canvas[j][k][0] = min(255, canvas[j][k][0] - superposition[j][k])
                cv2.imshow(db.get_img_path(i), canvas)
                cv2.waitKey(0)
                if save_file is not None:
                    cv2.imwrite(save_file, img)

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
