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
import os
import pprint
import time

import caffe
from utils.rap_db import RAPDataset
from wma_net.config import config, config_from_file, config_from_list
from wma_net.test import test_net


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='train WMA Network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device ID to use (default: 0)',
                        default=0, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default='./models/VGG_CNN_S/test_net.prototxt', type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default='./data/models/VGG_CNN_S.caffemodel', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--dbpath', dest='db_path',
                        help='the path of the RAP database',
                        default='./data/RAP', type=str)
    parser.add_argument('--setid', dest='par_set_id',
                        help='the index of training and testing data partition set',
                        default='0', type=int)
    parser.add_argument('--outputdir', dest='output_dir',
                        help='the directory to save outputs',
                        default='./output', type=str)

    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit()

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.config_file is not None:
        config_from_file(args.config_file)
    if args.set_configs is not None:
        config_from_list(args.set_configs)

    config.GPU_ID = args.gpu_id

    print('Using config:')
    pprint.pprint(config)

    while not os.path.exists(args.caffemodel) and args.wait:
        print('Waiting for {} to exist...'.format(args.caffemodel))
        time.sleep(10)

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]

    """Load RAP dataset"""
    db = RAPDataset(args.db_path, args.par_set_id)

    test_net(net, db, args.output_dir)
