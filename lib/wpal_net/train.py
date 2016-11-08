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

import os

import caffe
import google.protobuf as pb2
from caffe.proto import caffe_pb2
from utils.timer import Timer

from config import cfg
from test import test_net


class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over the snapshotting process.
    """

    def __init__(self, solver_prototxt, db, output_dir, do_flip,
                 snapshot_path=None):
        """Initialize the SolverWrapper."""
        self._output_dir = output_dir
        self._solver = caffe.SGDSolver(solver_prototxt)

        self._solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self._solver_param)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        self._snapshot_prefix = self._solver_param.snapshot_prefix + infix + '_iter_'

        if snapshot_path is not None:
            print ('Loading snapshot weights from {:s}').format(snapshot_path)
            self._solver.net.copy_from(snapshot_path)

            snapshot_path = snapshot_path.split('/')[-1]
            if snapshot_path.startswith(self._snapshot_prefix):
                print 'Warning! Existing snapshots may be overriden by new snapshots!'
 
        self._db = db
        self._solver.net.layers[0].set_db(self._db, do_flip)

    def snapshot(self):
        """Take a snapshot of the network."""
        net = self._solver.net

        filename = self._snapshot_prefix + ('{:d}'.format(self._solver.iter) + '.caffemodel')
        filepath = os.path.join(self._output_dir, filename)

        print 'Attempting to save snapshot to \"{}\"'.format(filepath)
        if not os.path.exists(self._output_dir):        
            os.makedirs(self._output_dir)
        net.save(str(filepath))
        print 'Wrote snapshot to: {:s}'.format(filepath)

        return filepath

    def train_model(self, max_iters):
        """Network training loop."""
        last_snapshot_iter = -1
        timer = Timer()
        model_paths = []
        while self._solver.iter < max_iters:
            # Make one SGD update
            timer.tic()
            self._solver.step(1)
            timer.toc()
            if self._solver.iter % (10 * self._solver_param.display) == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time)
            if self._solver.iter % 10 == 0:
                print "Python: iter", self._solver.iter
            if self._solver.iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = self._solver.iter
                model_paths.append(self.snapshot())

            if self._solver.iter % cfg.TRAIN.TEST_ITERS == 0:
                test_net(self._solver.test_net, self._db, self._output_dir)

        if last_snapshot_iter != self._solver.iter:
            model_paths.append(self.snapshot())
        return model_paths


def train_net(solver_prototxt, db, output_dir,
              snapshot_path=None, max_iters=40000):
    """Train a WMA network."""
    sw = SolverWrapper(solver_prototxt, db, output_dir, cfg.TRAIN.DO_FLIP,
                       snapshot_path=snapshot_path)

    print 'Solving...'
    model_paths = sw.train_model(max_iters)
    print 'done solving'
    return model_paths
