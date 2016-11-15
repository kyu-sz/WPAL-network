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

import os.path as osp

import numpy as np
import scipy.io as sio

import evaluate


class PETA:
    """This tool requires the PETA to be processed into similar form as RAP."""

    def __init__(self, db_path, par_set_id):
        self._db_path = db_path

        try:
            self.labels = sio.loadmat(osp.join(self._db_path, 'attributeLabels.mat'))['DataLabel']
        except NotImplementedError:
            import h5py
            print h5py.File(osp.join(self._db_path, 'attributeLabels.mat')).keys()
            self.labels = np.array(h5py.File(osp.join(self._db_path, 'attributeLabels.mat'))['DataLabel']).transpose()

        try:
            self.name = sio.loadmat(osp.join(self._db_path, 'attributesName.mat'))['attributesName']
        except NotImplementedError:
            import h5py
            print h5py.File(osp.join(self._db_path, 'attributesName.mat')).keys()
            self.name = h5py.File(osp.join(self._db_path, 'attributesName.mat'))['attributesName']

        try:
            self._partition = sio.loadmat(osp.join(self._db_path, 'partition.mat'))['partition']
        except NotImplementedError:
            import h5py
            print h5py.File(osp.join(self._db_path, 'partition.mat')).keys()
            self.name = h5py.File(osp.join(self._db_path, 'partition.mat'))['partition']

        self.num_attr = self.name.shape[0]
        self.test_ind = None
        self.train_ind = None
        self.label_weight = None
        self.set_partition_set_id(par_set_id)
        self.attr_group = [range(0, 4)]
        self.flip_attr_pairs = []  # The PETA database has no symmetric attribute pairs.

        self.expected_loc_centroids = np.ones(self.num_attr) * 2

    def evaluate_mA(self, attr, inds):
        return evaluate.mA(attr, self.labels[inds])

    def evaluate_example_based(self, attr, inds):
        return evaluate.example_based(attr, self.labels[inds])

    def set_partition_set_id(self, par_set_id):
        #self.train_ind = self._partition[par_set_id][0][0][0][0][0] - 1
        #self.test_ind = self._partition[par_set_id][0][0][0][1][0] - 1
        self.train_ind = filter(lambda x:x%5!=par_set_id, xrange(self.labels.shape[0]))
        self.test_ind = filter(lambda x:x%5==par_set_id, xrange(self.labels.shape[0]))
        
        pos_cnt = sum(self.labels[self.train_ind])
        self.label_weight = pos_cnt / len(self.train_ind)

    def get_img_path(self, img_id):
        return osp.join(self._db_path, 'Data', str(img_id + 1) + '.png')


if __name__ == '__main__':
    db = PETA('data/dataset/ProcessedPeta', 1)
    print "Labels:", db.labels.shape
    print len(db.train_ind)
    print len(db.test_ind)
    print 'Max training index: ', max(db.train_ind)
    print db.get_img_path(0)
    print db.num_attr
    print db.train_ind
    print db.test_ind
    print db.label_weight
