#!/usr/bin/env python

import _init_paths

from utils.rap_db import RAPDataset
import argparse
import cPickle


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='train WMA Network')
    parser.add_argument('--pkl', dest='pkl',
                        help='Saved attributes.',
                        default='./output/attributes.pkl', type=str)
    parser.add_argument('--set', dest='set_configs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--dbpath', dest='db_path',
                        help='the path of the RAP database',
                        default='./data/RAP', type=str)
    parser.add_argument('--setid', dest='par_set_id',
                        help='the index of training and testing data partition set',
                        default='0', type=int)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    """Load RAP dataset"""
    db = RAPDataset(args.db_path, args.par_set_id)

    f = open(args.pkl, 'rb')
    attr = cPickle.load(f)

    print attr.shape
