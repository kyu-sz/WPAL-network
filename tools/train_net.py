#!/usr/bin/env python

# --------------------------------------------------------------------
# This file is part of AM Network.
# 
# AM Network is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# AM Network is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with AM Network.  If not, see <http://www.gnu.org/licenses/>.
# --------------------------------------------------------------------

import __init__

import cv2
import caffe
import argparse
import pprint
import numpy as np
import sys

from utils.timer import Timer
from am_net.train import train_net


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='train AM Net')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device ID to use (default: -1 meaning using CPU only)',
                        default=-1, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default='./models/vgg_cnn_s/solver.prototxt', type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of training iterations',
                        default=40000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default='./models/vgg_cnn_s/vgg_cnn_s.caffemodel', type=str)
    parser.add_argument('--dbpath', dest='db_path',
                        help='the path of the RAP database',
						default='~/datasets/rap', type=str)
    parser.add_argument('--outputdir', dest='output_dir',
                        help='the directory to save outputs',
                        default='./output', type=str)

    #if len(sys.argv) == 1:
    #    parser.print_help()
    #    sys.exit()

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    # set up caffe
    if args.gpu_id == -1:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)

    print 'Output will be saved to `{:s}`'.format(args.output_dir)

    train_net(args.solver, args.db_path, args.output_dir,
        pretrained_model=args.pretrained_model,
		max_iters=args.max_iters)