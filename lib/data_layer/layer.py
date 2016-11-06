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

"""The data layer used during training to train a WPAL-net.
DataLayer implements a Caffe Python layer.
"""

from multiprocessing import Process, Queue

import numpy as np
from data_layer.minibatch import get_minibatch
from wpal_net.config import cfg

import caffe


class DataLayer(caffe.Layer):
    """WPAL-net data layer used for training."""

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next mini-batch.
        """
        return self._blob_queue.get()

    def set_db(self, db, do_flip):
        """Set the database to be used by this layer during training."""
        self._db = db

        """Enable prefetch."""
        self._blob_queue = Queue(32)
        self._prefetch_process = BlobFetcher(self._blob_queue, self._db, do_flip)
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
        top[idx].reshape(cfg.TRAIN.BATCH_SIZE, 3, max(cfg.TRAIN.SCALES), max(cfg.TRAIN.SCALES))
        self._name_to_top_map['data'] = idx
        idx += 1

        top[idx].reshape(cfg.TRAIN.BATCH_SIZE, cfg.NUM_ATTR)
        self._name_to_top_map['attr'] = idx
        idx += 1

        top[idx].reshape(cfg.TRAIN.BATCH_SIZE, cfg.NUM_ATTR)
        self._name_to_top_map['weight'] = idx
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

    def __init__(self, queue, db, do_flip=True):
        super(BlobFetcher, self).__init__()
        self._queue = queue
        self._db = db
        self._perm = None
        self._cur = 0
        self._do_flip = do_flip
        self._train_ind = self._db.train_ind
        self._weight = db.label_weight

        self._shuffle_train_inds()

        # fix the random seed for reproducibility
        np.random.seed(cfg.RNG_SEED)

    def _shuffle_train_inds(self):
        """Randomly permute the training database."""
        self._perm = np.random.permutation(xrange(len(self._train_ind * (2 if self._do_flip else 1))))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if self._cur >= len(self._db.train_ind):
            self._shuffle_train_inds()

        minibatch_inds = self._perm[self._cur:self._cur + cfg.TRAIN.BATCH_SIZE]
        self._cur += cfg.TRAIN.BATCH_SIZE
        return minibatch_inds

    def run(self):
        print 'BlobFetcher started'
        while True:
            minibatch_inds = self._get_next_minibatch_inds()
            minibatch_img_paths = \
                [self._db.get_img_path(self._db.train_ind[
                    i if i < len(self._db.train_ind) else i - len(self._db.train_ind)])
                 for i in minibatch_inds]
            minibatch_labels = \
                [self._db.labels[self._db.train_ind[
                     i if i < len(self._db.train_ind) else i - len(self._db.train_ind)]]
                 for i in minibatch_inds]
            minibatch_flip = \
                [0 if i < len(self._db.train_ind) else 1
                 for i in minibatch_inds]
            blobs = get_minibatch(minibatch_img_paths, minibatch_labels, minibatch_flip, self._db.flip_attr_pairs, self._weight)
            self._queue.put(blobs)
