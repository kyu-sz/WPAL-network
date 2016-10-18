#!/usr/bin/env python

# --------------------------------------------------------------------
# This file is part of WMA Network.
#
# WMA Network is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# WMA Network is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with WMA Network.  If not, see <http://www.gnu.org/licenses/>.
# --------------------------------------------------------------------

import os.path as osp

import numpy as np
import scipy.io as sio


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

		self.num_attrs = self.name.shape[0]
		self.test_ind = None
		self.train_ind = None
		self.set_partition_set_id(par_set_id)
     
                self.attr_group = [range(0,4)]

		self.flip_attr_pairs = []  # The PETA database has no symmetric attribute pairs.

	def evaluate_mA(self, attr, inds):
                print inds
		num = attr.__len__()
		gt = self.labels[inds]

		for i in xrange(self.num_attrs):
			print '--------------------------------------------'
			print i
			print sum([attr[j][i] * gt[j][i] for j in xrange(num)]) / sum([gt[j][i] for j in xrange(num)])
			print sum([(1 - attr[j][i]) * (1 - gt[j][i]) for j in xrange(num)]) / sum(
				[(1 - gt[j][i]) for j in xrange(num)])

		mA = (sum([(
			           sum([attr[j][i] * gt[j][i] for j in xrange(num)])
			           / sum([gt[j][i] for j in xrange(num)])
			           + sum([(1 - attr[j][i]) * (1 - gt[j][i]) for j in xrange(num)])
			           / sum([(1 - gt[j][i]) for j in xrange(num)])
		           ) for i in xrange(self.num_attrs)])) / (2 * self.num_attrs)

		return mA

	def set_partition_set_id(self, par_set_id):
		num_samples = self.labels.shape[0]
		block_size = num_samples / 5
		test_start = block_size * par_set_id
		test_end = block_size + test_start
		self.test_ind = range(test_start, test_end)
		self.train_ind = range(0, test_start) + range(test_end, num_samples)

	def get_img_path(self, img_id):
		return osp.join(self._db_path, 'Data', str(img_id + 1) + '.png')


if __name__ == '__main__':
	db = PETA('/home/ken.yu/databases/ProcessedPeta', 1)
	print "Labels:", db.labels.shape
	print db.train_ind.__len__()
	print db.test_ind.__len__()
	print 'Max training index: ', max(db.train_ind)
	print db.get_img_path(0)
	print db.num_attrs
