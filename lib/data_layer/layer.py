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

"""The data layer used during training to train a WMA-net.
DataLayer implements a Caffe Python layer.
"""

import caffe
import numpy as np
import yaml
from multiprocessing import Process, Queue

from wma_net.config import config
from data_layer.minibatch import get_minibatch

class DataLayer(caffe.Layer):
    """WMA-net data layer used for training."""

    def _shuffle_db_inds(self):
        """Randomly permute the training roidb."""
        self._perm = np.random.permutation(np.arange(len(self.train_ind)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if self._cur + config.TRAIN.IMS_PER_BATCH >= len(self.train_ind):
            self._shuffle_db_inds()

        db_inds = self._perm[self._cur:self._cur + config.TRAIN.IMS_PER_BATCH]
        self._cur += config.TRAIN.IMS_PER_BATCH
        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.
        """
        return self._blob_queue.get()

    def set_db(self, db):
        """Set the roidb to be used by this layer during training."""
        self._db = db
        self._image_names = self._db.img_names[self._db.train_ind]
        self._labels = self._db.labels[self._db.train_ind]

        self._shuffle_db_inds()

        """Enable prefetch."""
        self._blob_queue = Queue(10)
        self._prefetch_process = BlobFetcher(self._blob_queue, self._db)
        self._prefetch_process.start()
        # Terminate the child process when the parent exists
        def cleanup():
            print 'Terminating BlobFetcher'
            self._prefetch_process.terminate()
            self._prefetch_process.join()
        import atexit
        atexit.register(cleanup)

    def setup(self, bottom, top):
        """Setup the DataLayer."""

        self._name_to_top_map = {}

        # data blob: holds a batch of N images, each with 3 channels
        idx = 0
        top[idx].reshape(config.TRAIN.BATCH_SIZE, 3,
            max(config.TRAIN.SCALES), config.TRAIN.MAX_SIZE)
        self._name_to_top_map['data'] = idx
        idx += 1

        top[idx].reshape(config.TRAIN.BATCH_SIZE, config.NUM_ATTR)
        self._name_to_top_map['attr'] = idx
        idx += 1

        print 'DataLayer: name_to_top:', self._name_to_top_map
        assert len(top) == len(self._name_to_top_map)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()

        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

class BlobFetcher(Process):
    """Experimental class for prefetching blobs in a separate process."""
    def __init__(self, queue, db):
        super(BlobFetcher, self).__init__()
        self._queue = queue
        self._db = db
        self._perm = None
        self._cur = 0
        self._shuffle_db_inds()
        # fix the random seed for reproducibility
        np.random.seed(config.RNG_SEED)

    def _shuffle_db_inds(self):
        """Randomly permute the training roidb."""
        # TODO(rbg): remove duplicated code
        self._perm = np.random.permutation(np.arange(len(self._db.train_ind)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        # TODO(rbg): remove duplicated code
        if self._cur + config.TRAIN.IMS_PER_BATCH >= len(self._db.train_ind):
            self._shuffle_db_inds()

        db_inds = self._perm[self._cur:self._cur + config.TRAIN.IMS_PER_BATCH]
        self._cur += config.TRAIN.IMS_PER_BATCH
        return db_inds

    def run(self):
        print 'BlobFetcher started'
        while True:
            db_inds = self._get_next_minibatch_inds()
            minibatch_img_names = [self._db.img_names[i] for i in db_inds]
            minibatch_labels = [self._db.labels[i] for i in db_inds]
            blobs = get_minibatch(minibatch_img_names, minibatch_labels)
            self._queue.put(blobs)