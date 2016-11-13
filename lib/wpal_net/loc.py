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
from config import cfg


def gaussian_filter(shape, center_y, center_x, var=1):
    filter_map = np.ndarray(shape)
    for i in xrange(0, shape[0]):
        for j in xrange(0, shape[1]):
            filter_map[i][j] = math.exp(-(math.pow(i - center_y, 2) + math.pow(j - center_x, 2)) / 2 / var)
    return filter_map


def zero_mask(size, area):
    mask = np.zeros(size)
    for i in xrange(int(math.floor(area['y'])), min(size[0], int(math.ceil(area['y'] + area['h'])))):
        mask[i][int(math.floor(area['x'])):min(size[1], int(math.ceil(area['x'] + area['w'])))] += 1
    return mask


def localize(net, db, output_dir, pos_ave, neg_ave, dweight, attr_id=-1, vis=False, save_dir=None):
    """Test localization of a WPAL Network."""

    cfg.TEST.MAX_AREA = int(cfg.TEST.MAX_AREA / 2)

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

    dweight = np.log(dweight)
    weight_threshold = [sorted(x, reverse=1)[512] for x in dweight]

    num_bin_per_detector = []
    num_bin_per_layer = []
    for layer in cfg.LOC.LAYERS:
        bin_cnt = 0
        for level in layer.LEVELS:
            bin_cnt += level[0] * level[1]
        num_bin_per_detector.append(bin_cnt)
        num_bin_per_layer.append(bin_cnt * layer.NUM_DETECTOR)

    # find the index of layer the bin belongs to, given a global index of a bin
    def find_layer_ind(bin_ind):
        for i in xrange(len(num_bin_per_layer)):
            if bin_ind < num_bin_per_layer[i]:
                return i
            bin_ind -= num_bin_per_layer[i]

    def locate_bin_in_layer(bin_ind):
        layer_ind = layer_inds[bin_ind]
        """return: level_ind, detector_ind, bin_y, bin_x"""
        layer = cfg.LOC.LAYERS[layer_ind]
        for i in xrange(layer_ind):
            bin_ind -= num_bin_per_layer[i]
        for i in xrange(len(layer.LEVELS)):
            level = layer.LEVELS[i]
            if bin_ind >= level[0] * level[1] * layer.NUM_DETECTOR:
                bin_ind -= level[0] * level[1] * layer.NUM_DETECTOR
            else:
                return i, bin_ind / (level[0] * level[1]),\
                       bin_ind % (level[0] * level[1]) / level[1], bin_ind % level[1]

    cnt = 0
    for img_ind in db.test_ind:
        img_path = db.get_img_path(img_ind)
        name = os.path.split(img_path)[1]
        if attr_id != -1 and db.labels[img_ind][attr_id] == 0:
            print 'Image {} skipped for it is a negative sample for attribute {}!' \
                .format(name, db.attr_eng[attr_id][0][0])
            continue

        # check directory for saving visualization images
        vis_img_dir = os.path.join(output_dir, 'vis', 'body' if attr_id == -1 else db.attr_eng[attr_id][0][0], name)
        if not os.path.exists(vis_img_dir):
            os.makedirs(vis_img_dir)

        # prepare the image
        img = cv2.imread(img_path)
        print img.shape[0], img.shape[1]

        # pass the image throught the test net.
        attr, heat3, heat4, heat5, score, img_scale = recognize_attr(net, img, db.attr_group, threshold)
        all_attrs[cnt] = attr
        cnt += 1

        img_height = int(img.shape[0] * img_scale)
        img_width = int(img.shape[1] * img_scale)
        img = cv2.resize(img, (img_width, img_height))
        img_area = img_height * img_width
        cross_len = math.sqrt(img_area) * 0.05

        if attr_id != -1 and attr[attr_id] != 1:
            print 'Image {} skipped for failing to be recognized attribute {} from!' \
                .format(name, db.attr_eng[attr_id][0][0])
            continue

        layer_inds = [find_layer_ind(x) for x in xrange(len(score))]

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
            layer_ind = layer_inds[bin_ind]
            heat_ind = 0
            for i in xrange(layer_ind):
                heat_ind += cfg.LOC.LAYERS[i].NUM_DETECTOR
            _, detector_ind, _, _ = locate_bin_in_layer(bin_ind)
            return heat_maps[heat_ind + detector_ind]

        bin2heat = [find_heat_map(x) for x in xrange(len(score))]

        def get_effect_area(bin_ind):
            layer_ind = layer_inds[bin_ind]
            heat = bin2heat[bin_ind]
            level_ind, _, y, x = locate_bin_in_layer(bin_ind)
            layer = cfg.LOC.LAYERS[layer_ind]
            level = layer.LEVELS[level_ind]
            bin_h = heat.shape[0] * (1 + layer.OVERLAP[0] * (level[0] - 1)) / level[0]
            bin_w = heat.shape[1] * (1 + layer.OVERLAP[1] * (level[1] - 1)) / level[1]
            y_start = (1 - layer.OVERLAP[0]) * y * bin_h
            x_start = (1 - layer.OVERLAP[1]) * x * bin_h
            return {'y': y_start, 'x': x_start, 'h': bin_h, 'w': bin_w}

        # find the target a bin detects.
        # TODO: check the target location against the bin's region
        def find_target(bin_ind):
            layer_ind = layer_inds[bin_ind]
            effect_area = get_effect_area(bin_ind)
            locs = np.where(bin2heat[bin_ind] == score[bin_ind])
            if len(locs[0]) == 0:
                print 'Cannot find max value {} of bin {}'.format(score[bin_ind], bin_ind)
                return [0]
            if len(locs[0]) > 1:
                for i in xrange(len(locs[0])):
                    loc = [locs[0][i], locs[1][i]]
                    if effect_area['y'] <= loc[0] < effect_area['y'] + effect_area['h']\
                            and effect_area['x'] <= loc[1] < effect_area['x'] + effect_area['w']:
                        return loc[0] + 0.5, loc[1] + 0.5
            return locs[0][0] + 0.5, locs[1][0] + 0.5

        # find all the targets in advance
        target = [find_target(j) for j in xrange(len(score))]

        total_superposition = np.zeros_like(img, dtype=float)
        for a in attr_list:
            # calc the actual contribution weights
            def w_func(x):
                return 0\
                    if dweight[a][x] < weight_threshold[a]\
                    else score[x] / (pos_ave[a][x] if attr[a] else neg_ave[a][x]) * dweight[a][x]
            w_sum = sum([w_func(j) for j in xrange(len(score))])

            # Center of the feature.
            center_y = sum([w_func(j) / w_sum * target[j][0] / bin2heat[j].shape[0]
                     for j in xrange(len(score))])
            center_x = sum([w_func(j) / w_sum * target[j][1] / bin2heat[j].shape[1]
                     for j in xrange(len(score))])
            # Superposition of the heat maps.
            superposition = sum([cv2.resize(w_func(j) / w_sum * bin2heat[j].astype(float)
                                            * gaussian_filter(bin2heat[j].shape,
                                                              center_y * bin2heat[j].shape[0],
                                                              center_x * bin2heat[j].shape[1],
                                                              img_area / bin2heat[j].shape[0] * bin2heat[j].shape[1])
                                            * zero_mask(bin2heat[j].shape, get_effect_area(j)),
                                            (img_width, img_height))
                                 for j in xrange(len(score))])

            mean = (superposition.max() + superposition.min()) / 2
            val_range = superposition.max() - superposition.min()
            superposition = (superposition - mean) * 255 / val_range / len(attr_list)
            for j in xrange(img_height):
                for k in xrange(img_width):
                    total_superposition[j][k][2] += superposition[j][k]
                    total_superposition[j][k][1] -= superposition[j][k]
                    total_superposition[j][k][0] -= superposition[j][k]
            cv2.line(total_superposition,
                     (int(img_width * center_x - cross_len), int(img_height * center_y)),
                     (int(img_width * center_x + cross_len), int(img_height * center_y)),
                     (0, 255, 255))
            cv2.line(total_superposition,
                     (int(img_width * center_x), int(img_height * center_y - cross_len)),
                     (int(img_width * center_x), int(img_height * center_y + cross_len)),
                     (0, 255, 255))

            if attr_id != -1:
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
                    val_scale = 255.0 / max(max(__) for __ in bin2heat[j])
                    canvas = np.zeros_like(img)
                    canvas[..., 2] = cv2.resize((bin2heat[j] * val_scale).astype('uint8'),
                                                (img.shape[1], img.shape[0]))
                    _y = 1.0 * target[j][0] / bin2heat[j].shape[0]
                    _x = 1.0 * target[j][1] / bin2heat[j].shape[1]
                    cv2.line(canvas,
                             (int(img_width * _x - cross_len), int(img_height * _y)),
                             (int(img_width * _x + cross_len), int(img_height * _y)),
                             (0, 255, 255))
                    cv2.line(canvas,
                             (int(img_width * _x), int(img_height * _y - cross_len)),
                             (int(img_width * _x), int(img_height * _y + cross_len)),
                             (0, 255, 255))
                    cv2.imshow("Masked heat map", canvas)
                    cv2.waitKey(0)

                    print 'Saving to:', os.path.join(vis_img_dir, 'final.jpg')
                    cv2.imwrite(os.path.join(vis_img_dir, 'heat{}.jpg'.format(j)),
                                canvas)

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
        print 'Saving to:', os.path.join(vis_img_dir, 'final.jpg')
        cv2.imwrite(os.path.join(vis_img_dir, 'final.jpg'), canvas)

    cv2.destroyWindow("heat_maps")
    cv2.destroyWindow("sup")
    cv2.destroyWindow("img")

if __name__ == '__main__':
    print gaussian_filter((8, 3), 2, 1)
