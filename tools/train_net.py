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

import _init_path

import argparse

from utils.timer import Timer
from wma_net.train import train_net
import caffe
from utils.rap_db import RAPDataset
from wma_net.config import config, config_from_file, config_from_list
from wma_net.train import train_net


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='train WMA Network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device ID to use (default: -1 meaning using CPU only)',
                        default=-1, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default='./models/VGG_CNN_S/solver.prototxt', type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of training iterations',
                        default=40000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default='./data/pretrained_models/VGG_CNN_S.caffemodel', type=str)
    parser.add_argument('--dbpath', dest='db_path',
                        help='the path of the RAP database',
                        default='./data/RAP', type=str)
    parser.add_argument('--setid', dest='par_set_id',
                        help='the index of training and testing data partition set',
                        default='0', type=int)
    parser.add_argument('--outputdir', dest='output_dir',
                        help='the directory to save outputs',
                        default='./output', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)

    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit()

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        config_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        config_from_list(args.set_cfgs)

    config.GPU_ID = args.gpu_id

    # set up Caffe
    if args.gpu_id == -1:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)

    print 'Output will be saved to `{:s}`'.format(args.output_dir)

    """Load RAP dataset"""
    db = RAPDataset(args.db_path, args.par_set_id)

    train_net(args.solver, db, args.output_dir,
              pretrained_model=args.pretrained_model,
              max_iters=args.max_iters)
