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

from recog import recognize_attr
from wpal_net.config import cfg


def gaussian_filter(shape, center_y, center_x, var=1):
    var = var * var * 100
    filter_map = np.ndarray(shape)
    for i in xrange(0, shape[0]):
        for j in xrange(0, shape[1]):
            filter_map[i][j] = math.exp(-(math.pow(i - center_y, 2) + math.pow(j - center_x, 2)) / 2 / var)
    return filter_map


def zero_mask(size, y, x, h, w):
    mask = np.zeros(size)
    for i in xrange(y, y + h):
        mask[x:x+w] += 1
    return mask


def localize(net, db, output_dir, ave, sigma, dweight, attr_id=-1, vis=False, save_dir=None):
    """Test localization of a WPAL Network."""

    num_images = len(db.test_ind)

    all_attrs = [[] for _ in xrange(num_images)]

    threshold = np.ones(db.num_attr) * 0.5;

    if attr_id == -1:
        # localize whole body outline
        attr_list = xrange(db.num_attr)
    else:
        # localize only one attribute
        attr_list = []
        attr_list.append(attr_id)

    weight_threshold = [sorted(center_x,reverse=1)[64] for center_x in dweight]

    num_bin_per_detector = []
    num_bin_per_layer = []
    for layer in cfg.LOC.LAYERS:
        bin_cnt = 0
        for num_bin_per_level in layer.NUM_BIN:
            bin_cnt += num_bin_per_level[0] * num_bin_per_level[1]
        num_bin_per_detector.append(bin_cnt)
        num_bin_per_layer.append(bin_cnt * layer.NUM_DETECTOR)

    # find the index of layer the bin belongs to, given a global index of a bin
    def find_layer_ind(bin_ind):
        cnt = 0
        for layer_ind_iter in xrange(len(num_bin_per_layer)):
            if bin_ind < num_bin_per_layer[layer_ind_iter] + cnt:
                return layer_ind_iter,
            cnt += num_bin_per_layer[layer_ind_iter]

    cnt = 0
    for img_ind in db.test_ind:
        img_path = db.get_img_path(img_ind)
        name = os.path.split(img_path)[1]
        if attr_id != -1 and db.labels[img_ind][attr_id] == 0:
            print 'Image {} skipped for it is a negative sample for attribute {}!' \
                .format(name, db.attr_eng[attr_id][0][0])
            continue

        # prepare the image
        img = cv2.imread(img_path)
        img_width = img.shape[1] * cfg.TEST.SCALE / img.shape[0]
        img_height = cfg.TEST.SCALE
        img = cv2.resize(img, (img_width, img_height))
        img_area = img_height * img_width
        cross_len = math.sqrt(img_area) * 0.05

        # pass the image throught the test net.
        attr, heat3, heat4, heat5, score = recognize_attr(net, img, db.attr_group, threshold)
        all_attrs[cnt] = attr
        cnt += 1

        if attr_id != -1 and attr[attr_id] != 1:
            print 'Image {} skipped for failing to be recognized attribute {} from!' \
                .format(name, db.attr_eng[attr_id][0][0])
            continue

        # concat the heat maps before the detectors
        heat_maps = []
        for hm in heat3:
            heat_maps.append(hm)
        for hm in heat4:
            heat_maps.append(hm)
        for hm in heat5:
            heat_maps.append(hm)

        # find heat map of a bin
        def find_heat_map(bin_ind):
            return heat_maps[bin_ind / num_bin_per_detector[find_layer_ind(bin_ind)]]

        def get_effect_area(bin_ind):
            layer_ind = find_layer_ind(bin_ind)
            for j in xrange(layer_ind):
                bin_ind -= num_bin_per_layer[j]
            bin_ind %= num_bin_per_detector[layer_ind]
            l = cfg.LOC.LAYERS[layer_ind]
            for num_bin_per_level in l.NUM_BIN:
                if bin_ind >= num_bin_per_level[0] * num_bin_per_level[1]:
                    bin_ind -= num_bin_per_level[0] * num_bin_per_level[1]
                else:
                    heat = find_heat_map(bin_ind)
                    y = bin_ind / num_bin_per_level[0]
                    x = bin_ind % num_bin_per_level[0]
                    bin_h = heat.shape[0] * (1 + layer.OVERLAP[0] * (num_bin_per_level[0] - 1)) / num_bin_per_level[0]
                    bin_w = heat.shape[1] * (1 + layer.OVERLAP[1] * (num_bin_per_level[1] - 1)) / num_bin_per_level[1]
                    y_start = (1 - layer.OVERLAP[0]) * y * bin_h
                    x_start = (1 - layer.OVERLAP[1]) * x * bin_h
                    return y_start, x_start, bin_h, bin_w

        # find the target a bin detects.
        # TODO: check the target location against the bin's region
        def find_target(bin_ind, effect_area):
            locs = np.where(find_heat_map(bin_ind) == score[bin_ind])
            if len(locs[0]) > 1:
                for loc in locs:
                    if effect_area[0] <= loc[0] < effect_area[0] + effect_area[2]\
                        and effect_area[1] <= loc[1] < effect_area[1] + effect_area[3]:
                        return loc
                return locs[0]
            else:
                return locs

        # find all the targets in advance
        target = [find_target(j)[:] for j in xrange(len(score))]

        total_superposition = np.zeros_like(img, dtype=float)
        for a in attr_list:
            # calc the actual contribution weights
            w_func = lambda center_x: 0\
                if dweight[a][center_x] < weight_threshold[a]\
                else score[center_x] * dweight[a][center_x]
            w_sum = sum([w_func(j) for j in xrange(len(score))])

            # Center of the feature.
            center_y = sum([w_func(j) / w_sum * target[j][0] / np.array(find_heat_map(j)).shape[0]
                     for j in xrange(len(score))])
            center_x = sum([w_func(j) / w_sum * target[j][1] / np.array(find_heat_map(j)).shape[1]
                     for j in xrange(len(score))])
            # Superposition of the heat maps.
            superposition = sum([cv2.resize(w_func(j) / w_sum * find_heat_map(j)
                                            * gaussian_filter(find_heat_map(j).shape,
                                                              center_y * find_heat_map(j).shape[0],
                                                              center_x * find_heat_map(j).shape[1],
                                                              img_area / find_heat_map(j).shape[0] * find_heat_map(j).shape[1])
                                            * zero_mask(find_heat_map(j).shape, get_effect_area(j)),
                                            (img_width, img_height))
                                 for j in xrange(len(score))])

            mean = (superposition.max() + superposition.min()) / 2
            val_range = superposition.max() - superposition.min()
            superposition = (superposition - mean) * 255 / val_range
            for j in xrange(img_height):
                for k in xrange(img_width):
                    total_superposition[j][k][2] += superposition[j][k]
                    total_superposition[j][k][1] -= superposition[j][k]
                    total_superposition[j][k][0] -= superposition[j][k]
            cv2.line(total_superposition, (img_width * center_x - cross_len, img_height * center_y), (img_width * center_x + cross_len, img_height * center_y), (0, 255, 255))
            cv2.line(total_superposition, (img_width * center_x, img_height * center_y - cross_len), (img_width * center_x, img_height * center_y + cross_len), (0, 255, 255))

            for j in xrange(img_height):
                for k in xrange(img_width):
                    total_superposition[j][k][2] = min(255, max(0, total_superposition[j][k][2]))
                    total_superposition[j][k][1] = min(255, max(0, total_superposition[j][k][1]))
                    total_superposition[j][k][0] = min(255, max(0, total_superposition[j][k][0]))
            cv2.imshow("sup", total_superposition.astype('uint8'))
            cv2.imshow("img", img)

            for j in [_[0] for _ in sorted(enumerate([w_func(k) for k in xrange(len(score))]),
                                           key=lambda x:x[1],
                                           reverse=1)][0:8]:
                print name, j, w_func(j)
                val_scale = 255.0 / max(max(__) for __ in find_heat_map(j))
                canvas = np.zeros_like(img)
                canvas[..., 2] = cv2.resize((find_heat_map(j) * val_scale).astype('uint8'), (img.shape[1], img.shape[0]))
                _y = 1.0 * target[j][0] / np.array(find_heat_map(j)).shape[0]
                _x = 1.0 * target[j][1] / np.array(find_heat_map(j)).shape[1]
                cv2.line(canvas, (img_width * _x - cross_len, img_height * _y), (img_width * _x + cross_len, img_height * _y), (0, 255, 255))
                cv2.line(canvas, (img_width * _x, img_height * _y - cross_len), (img_width * _x, img_height * _y + cross_len), (0, 255, 255))
                cv2.imshow("heat_maps", canvas)
                cv2.waitKey(0)

        canvas = total_superposition + img
        for j in xrange(img_height):
            for k in xrange(img_width):
                canvas[j][k][2] = min(255, max(0, canvas[j][k][2]))
                canvas[j][k][1] = min(255, max(0, canvas[j][k][1]))
                canvas[j][k][0] = min(255, max(0, canvas[j][k][0]))
        canvas = canvas.astype('uint8')

        if vis:
            cv2.imshow("img", canvas)
            cv2.waitKey(0)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, name), img)

    cv2.destroyWindow("heat_maps")
    cv2.destroyWindow("sup")
    cv2.destroyWindow("img")

if __name__ == '__main__':
    print gaussian_filter((8, 3), 2, 1)
