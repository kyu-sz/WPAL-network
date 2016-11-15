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
from scipy.cluster.vq import kmeans, vq
from scipy.misc import imresize
import pylab

from recog import recognize_attr
from config import cfg
from utils.kmeans import weighted_kmeans


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

def cluster_heat(img, k, stepsX, max_round=1000):
    """Return centroids of heat clusters (in x-y order)."""
    stepsY = stepsX * img.shape[0] / img.shape[1]

    thresh = max(np.mean(img), np.median(img), 0)

    dy = img.shape[0] / stepsY
    dx = img.shape[1] / stepsX

    act_points = []
    for y in xrange(stepsY):
        for x in xrange(stepsX):
            score = img[y * dy: (y + 1) * dy, x * dx: (x + 1) * dx]
            if score > thresh:
                act_points.append([x, y, score])
    act_points = np.array(act_points)

    centroids, clusterAssment = weighted_kmeans(act_points, 4, max_round)
    return centroids


def localize(net, db, output_dir, pos_ave, neg_ave, dweight, attr_id=-1, display=True, save_dir=None, max_count=-1):
    """Test localization of a WPAL Network."""

    cfg.TEST.MAX_AREA = int(cfg.TEST.MAX_AREA / 2)

    num_images = len(db.test_ind)
    if (max_count == -1):
        max_count = num_images

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
        """return: level_ind, detector_ind, bincentroids[1], bincentroids[0]"""
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

        # prepare the image
        img = cv2.imread(img_path)
        print img.shape[0], img.shape[1]

        # pass the image throught the test net.
        attr, heat3, heat4, heat5, score, img_scale = recognize_attr(net, img, db.attr_group, threshold)
        all_attrs[cnt] = attr

        img_height = int(img.shape[0] * img_scale)
        img_width = int(img.shape[1] * img_scale)
        img = cv2.resize(img, (img_width, img_height))
        img_area = img_height * img_width
        cross_len = math.sqrt(img_area) * 0.05

        if attr_id != -1 and attr[attr_id] != 1:
            print 'Image {} skipped for failing to be recognized attribute {} from!' \
                .format(name, db.attr_eng[attr_id][0][0])
            continue

        # check directory for saving visualization images
        vis_img_dir = os.path.join(output_dir, 'display', 'body' if attr_id == -1 else db.attr_eng[attr_id][0][0], name)
        if not os.path.exists(vis_img_dir):
            os.makedirs(vis_img_dir)

        if display:
            cv2.imshow("img", img)
            #cv2.waitKey(0)

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

        canvas = np.array(img)
        total_superposition = np.zeros(img.shape[0:2], dtype=float)
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

            thresh = min(np.median(superposition), np.mean(superposition))
            val_range = superposition.max() - superposition.min()
            superposition = (superposition - thresh) * 255 / val_range / len(attr_list)
            total_superposition += superposition

            expected_num_centroids = db.expected_loc_centroids[a]
            centroids = cluster_heat(superposition,
                                     expected_num_centroids + 2,
                                     img.shape[1],
                                     max_round=100 / len(attr_list))

            for c in centroids[:expected_num_centroids]:
                cv2.line(canvas,
                         (int(c[0] - cross_len), int(c[1])),
                         (int(c[0] + cross_len), int(c[1])),
                         (0, 255, 255))
                cv2.line(canvas,
                         (int(c[0]), int(c[1] - cross_len)),
                         (int(c[0]), int(c[1] + cross_len)),
                         (0, 255, 255))

            if attr_id != -1:
                for j in [_[0] for _ in sorted(enumerate([w_func(k) for k in xrange(len(score))]),
                                               key=lambda x:x[1],
                                               reverse=1)][0:8]:
                    print name, j, w_func(j)
                    val_scale = 255.0 / max(max(__) for __ in bin2heat[j])
                    heat_vis = np.zeros_like(img)
                    heat_vis[..., 2] = cv2.resize((bin2heat[j] * val_scale).astype('uint8'),
                                                  (img.shape[1], img.shape[0]))
                    y = 1.0 * target[j][0] / bin2heat[j].shape[0]
                    x = 1.0 * target[j][1] / bin2heat[j].shape[1]
                    cv2.line(heat_vis,
                             (int(img_width * x - cross_len), int(img_height * y)),
                             (int(img_width * x + cross_len), int(img_height * y)),
                             (0, 255, 255))
                    cv2.line(heat_vis,
                             (int(img_width * x), int(img_height * y - cross_len)),
                             (int(img_width * x), int(img_height * y + cross_len)),
                             (0, 255, 255))
                    if display:
                        cv2.imshow("heat", heat_vis)
                        cv2.waitKey(100)

                    print 'Saving to:', os.path.join(vis_img_dir, 'heat{}.jpg'.format(j))
                    cv2.imwrite(os.path.join(vis_img_dir, 'heat{}.jpg'.format(j)),
                                heat_vis)

            print 'Localized attribute {}: {}!'.format(a, db.attr_eng[a][0][0])

        for j in xrange(img_height):
            for k in xrange(img_width):
                canvas[j][k][2] = min(255, max(0, canvas[j][k][2] + max(0, total_superposition[j][k])))
                canvas[j][k][1] = min(255, max(0, canvas[j][k][1]))
                canvas[j][k][0] = min(255, max(0, canvas[j][k][0]))
        canvas = canvas.astype('uint8')

        if display:
            cv2.imshow("img", canvas)
            cv2.waitKey(0)
        print 'Saving to:', os.path.join(vis_img_dir, 'final.jpg')
        cv2.imwrite(os.path.join(vis_img_dir, 'final.jpg'), canvas)

        cnt += 1
        print 'Localized {} targets!'.format(cnt)
        if cnt >= max_count:
            break

    cv2.destroyWindow("heat")
    cv2.destroyWindow("img")

if __name__ == '__main__':
    print gaussian_filter((8, 3), 2, 1)
