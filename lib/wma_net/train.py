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

import os

import caffe
import google.protobuf as pb2
from caffe.proto import caffe_pb2
from utils.timer import Timer

from config import config
from test import test_net


class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over the snapshotting process.
    """

    def __init__(self, solver_prototxt, db, output_dir, do_flip,
                 pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.output_dir = output_dir
        self.solver = caffe.SGDSolver(solver_prototxt)

        if pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(pretrained_model)
            self.solver.net.copy_from(pretrained_model)

        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)

        self._db = db
        self.solver.net.layers[0].set_db(self._db, do_flip)

    def snapshot(self):
        """Take a snapshot of the network."""
        net = self.solver.net

        infix = ('_' + config.TRAIN.SNAPSHOT_INFIX
                 if config.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (self.solver_param.snapshot_prefix + infix +
                    '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
        filename = os.path.join(self.output_dir, filename)

        net.save(str(filename))
        print 'Wrote snapshot to: {:s}'.format(filename)

        return filename

    def train_model(self, max_iters):
        """Network training loop."""
        last_snapshot_iter = -1
        timer = Timer()
        model_paths = []
        while self.solver.iter < max_iters:
	    print "Python: iter", self.solver.iter
            # Make one SGD update
            timer.tic()
            self.solver.step(1)
            timer.toc()
            if self.solver.iter % (10 * self.solver_param.display) == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if self.solver.iter % config.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = self.solver.iter
                model_paths.append(self.snapshot())

            if self.solver.iter % config.TRAIN.TEST_ITERS == 0:
                test_net(self.solver.net, self._db, self._output_dir)

        if last_snapshot_iter != self.solver.iter:
            model_paths.append(self.snapshot())
        return model_paths


def train_net(solver_prototxt, db, output_dir,
              pretrained_model=None, max_iters=40000):
    """Train a WMA network."""
    sw = SolverWrapper(solver_prototxt, db, output_dir, config.TRAIN.DO_FLIP,
                       pretrained_model=pretrained_model)

    print 'Solving...'
    model_paths = sw.train_model(max_iters)
    print 'done solving'
    return model_paths
