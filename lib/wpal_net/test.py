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
from recog import recognize_attr
from wpal_net.config import cfg

def test_net(net, db, output_dir):
    """Test a Weakly-supervised Pedestrian Attribute Localization Network on an image database."""

    num_images = len(db.test_ind)

    all_attrs = [[] for _ in xrange(num_images)]

    # timers
    _t = {'recognize_attr' : Timer()}

    threshold = np.ones(db.num_attr) * 0.5;

    cnt = 0
    for i in db.test_ind:
        img_path = db.get_img_path(i)
        img = cv2.imread(img_path)
        _t['recognize_attr'].tic()
        attr, _, score, _ = recognize_attr(net, img, db.attr_group, threshold)
        _t['recognize_attr'].toc()
        all_attrs[cnt] = attr
        cnt += 1

        if cnt % 100 == 0:
            print 'recognize_attr: {:d}/{:d} {:.3f}s' \
                  .format(cnt, num_images, _t['recognize_attr'].average_time)

    attr_file = os.path.join(output_dir, 'attributes.pkl')
    with open(attr_file, 'wb') as f:
        cPickle.dump(all_attrs, f, cPickle.HIGHEST_PROTOCOL)

    mA, accPerAttr, challenging = db.evaluate_mA(all_attrs, db.test_ind)
    print 'mA={:f}'.format(mA)
    print 'Challenging attributes:', challenging

    acc, prec, rec, f1 = db.evaluate_example_based(all_attrs, db.test_ind)

    print 'Acc={:f} Prec={:f} Rec={:f} F1={:f}'.format(acc, prec, rec, f1)

    acc_file = os.path.join(output_dir, 'acc.txt')
    with open(acc_file, 'w') as f:
        for i in xrange(min(db.num_attr, cfg.TEST.MAX_NUM_ATTR)):
            f.write('{}: {}\n'.format(db.attr_eng[i][0][0], accPerAttr[i]))
        f.write('mA: {}\n'.format(mA))
        f.write('Acc: {} \t Prec: {} \t Rec: {} \t F1: {}\n'.format(acc, prec, rec, f1))

