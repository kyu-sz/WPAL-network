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

import _init_path

import argparse
import cPickle
import os
import pprint
import sys
import time
import numpy as np

import caffe
from wpal_net.config import cfg, cfg_from_file, cfg_from_list
from wpal_net.loc import test_localization, locate_in_video


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='test WPAL-network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device ID to use (default: -1)',
                        default=-1, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional cfg file', default=None, type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set cfg keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--db', dest='db',
                        help='the name of the database',
                        default=None, type=str)
    parser.add_argument('--setid', dest='par_set_id',
                        help='the index of training and testing data partition set',
                        default='0', type=int)
    parser.add_argument('--outputdir', dest='output_dir',
                        help='the directory to save outputs',
                        default='./output', type=str)
    parser.add_argument('--detector-weight', dest='dweight',
                        help='the cPickle file storing the weights of detectors',
                        default=None, type=str)
    parser.add_argument('--display', dest='display',
                        help='whether to display on screen',
                        default=1, type=int)
    parser.add_argument('--max-count', dest='max_count',
                        help='max number of images to perform localization',
                        default=-1, type=int)
    parser.add_argument('--attr-ids', dest='attr_id_list',
                        help='the IDs of the attributes to be located and visualized, '
                             'split by comma. '
                             '-1 for whole body shape. '
                             '-2 for all attributes. ',
                        default=-1, type=str)
    parser.add_argument('--video', dest='video',
                        help='specifying this argument means the program is to perform '
                             'attribute localization on a video but not the database pictures ',
                        default=None, type=str)
    parser.add_argument('--tracking-res', dest='tracking_res',
                        help='when conducting attribute localization on a video, '
                             'pedestrian tracking should be performed in advance, '
                             'and results are input as an input file.',
                        default=None, type=str)

    args = parser.parse_args()

    if args.prototxt is None or args.caffemodel is None or args.db is None or args.dweight is None:
        parser.print_help()
        sys.exit()

    return args


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.GPU_ID = args.gpu_id

    print('Using cfg:')
    pprint.pprint(cfg)

    while not os.path.exists(args.caffemodel) and args.wait:
        print('Waiting for {} to exist...'.format(args.caffemodel))
        time.sleep(10)

    if args.db == 'RAP':
        """Load RAP database"""
        from utils.rap_db import RAP

        db = RAP(os.path.join('data', 'dataset', args.db), args.par_set_id)
    else:
        """Load PETA dayanse"""
        from utils.peta_db import PETA

        db = PETA(os.path.join('data', 'dataset', args.db), args.par_set_id)

    f = open(args.dweight, 'rb')
    pack = cPickle.load(f)

    # set up Caffe
    if args.gpu_id == -1:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)

    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]

    if args.video is not None:
        locate_in_video(net,
                        db,
                        args.video, args.tracking_res,
                        args.output_dir,
                        pack['pos_ave'], pack['neg_ave'], pack['binding'],
                        args.attr_id_list)
    else:
        if args.attr_id_list == '-2':
            for a in xrange(db.num_attr):
                test_localization(net, db, args.output_dir, pack['pos_ave'], pack['neg_ave'], pack['binding'],
                                  attr_id=a,
                                  display=args.display,
                                  max_count=args.max_count)
            test_localization(net, db, args.output_dir, pack['pos_ave'], pack['neg_ave'], pack['binding'],
                              attr_id=-1,
                              display=args.display,
                              max_count=args.max_count)
        else:
            for attr_id in args.attr_id_list.split(','):
                test_localization(net, db, args.output_dir, pack['pos_ave'], pack['neg_ave'], pack['binding'],
                                  attr_id=int(attr_id),
                                  display=args.display,
                                  max_count=args.max_count)
