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

import math
import os

import cv2
import numpy as np
from utils.timer import Timer

from recog import recognize_attr
from wpal_net.config import cfg


def gaussian_filter(shape, y, x, var=1):
    filter_map = np.ndarray(shape)
    for i in xrange(0, shape[0]):
        for j in xrange(0, shape[1]):
            filter_map[i][j] = math.exp(-(math.pow(i - y, 2) + math.pow(j - x, 2)) / 2 / var)
    return filter_map


def localize(net, db, output_dir, ave, sigma, dweight, attr_id=-1, vis=False, save_dir=None):
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

    weight_threshold = [sorted(x,reverse=1)[64] for x in dweight]

    cnt = 0
    for i in db.test_ind:
        img_path = db.get_img_path(i)
        name = os.path.split(img_path)[1]
        if attr_id != -1 and db.labels[i][attr_id] == 0:
            print 'Image {} skipped for it is a negative sample for attribute {}!' \
                .format(name, db.attr_eng[attr_id][0][0])
            continue

        img = cv2.imread(img_path)
        w = img.shape[1] * cfg.TEST.SCALE / img.shape[0]
        h = cfg.TEST.SCALE
        img = cv2.resize(img, (w, h))
        img_area = h * w
        cross_len = math.sqrt(img_area) * 0.05

        _t['recognize_attr'].tic()
        attr, heat3, heat4, heat5, score = recognize_attr(net, img, db.attr_group, threshold)
        _t['recognize_attr'].toc()
        all_attrs[cnt] = attr
        cnt += 1

        if attr_id != -1 and attr[attr_id] != 1:
            print 'Image {} skipped for failing to be recognized attribute {} from!' \
                .format(name, db.attr_eng[attr_id][0][0])
            continue

        canvas = np.array(img, dtype=float)
        for attr in attr_list:
            w_func = lambda x: 0\
                if dweight[attr][x] < weight_threshold[attr]\
                else math.log(dweight[attr][x], math.e)
            w_sum = sum([w_func(j) for j in xrange(len(score))])
            get_heat_map = lambda x: heat3[x] if x < heat3.shape[0] else \
                heat4[x - heat3.shape[0]] if x - heat3.shape[0] < heat4.shape[0] else \
                    heat5[x - heat3.shape[0] - heat4.shape[0]]
            heat = [get_heat_map(j) for j in xrange(len(score))]

            def find_target(ind):
                locs = np.where(heat[ind] == score[ind])
                if len(locs[0]) > 1:
                    return locs[0][0], locs[0][1]
                else:
                    return locs

            target = [find_target(j)[:] for j in xrange(len(score))]
            # Center of the feature.
            x = sum([w_func(j) / w_sum * target[j][1] / np.array(heat[j]).shape[1]
                     for j in xrange(len(score))])
            y = sum([w_func(j) / w_sum * target[j][0] / np.array(heat[j]).shape[0]
                     for j in xrange(len(score))])
            # Superposition of the heat maps.
            superposition = sum([cv2.resize(w_func(j) / w_sum * heat[j]
                                            * gaussian_filter(heat[j].shape,
                                                              y * heat[j].shape[0],
                                                              x * heat[j].shape[1],
                                                              img_area / heat[j].shape[0] * heat[j].shape[1]),
                                            (w, h))
                                 for j in xrange(len(score))])

            mean = (superposition.max() + superposition.min()) / 2
            val_range = superposition.max() - superposition.min()
            superposition = (superposition - mean) * 255 / val_range
            for j in xrange(h):
                for k in xrange(w):
                    canvas[j][k][2] += superposition[j][k]
                    canvas[j][k][0] -= superposition[j][k]
            cv2.line(canvas, (w * x - cross_len, h * y), (w * x + cross_len, h * y), (0, 255, 255))
            cv2.line(canvas, (w * x, h * y - cross_len), (w * x, h * y + cross_len), (0, 255, 255))

            for j in xrange(h):
                for k in xrange(w):
                    canvas[j][k][2] = min(255, max(0, canvas[j][k][2]))
                    canvas[j][k][0] = min(255, max(0, canvas[j][k][0]))
            cv2.imshow("Loc", canvas.astype('uint8'))

            for j in [_[0] for _ in sorted(enumerate([w_func(k) for k in xrange(len(score))]),
                                           key=lambda x:x[1],
                                           reverse=1)][0:8]:
                print name, j, w_func(j)
                val_scale = 255 / max(max(__) for __ in heat[j])
                cv2.imshow("Heat", cv2.resize((heat[j] * val_scale).astype('uint8'), (img.shape[1], img.shape[0])))
                cv2.waitKey(0)

        for j in xrange(h):
            for k in xrange(w):
                canvas[j][k][2] = min(255, max(0, canvas[j][k][2]))
                canvas[j][k][0] = min(255, max(0, canvas[j][k][0]))
        canvas = canvas.astype('uint8')

        if vis:
            cv2.imshow("Loc", canvas)
            cv2.waitKey(0)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, name), img)

    cv2.destroyWindow("Loc")

if __name__ == '__main__':
    print gaussian_filter((8, 3), 2, 1)
