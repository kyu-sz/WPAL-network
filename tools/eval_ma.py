#!/usr/bin/env python

import _init_path

import argparse
import cPickle
import os


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
    parser.add_argument('--db', dest='db',
                        help='the name of the database',
                        default='RAP', type=str)
    parser.add_argument('--setid', dest='par_set_id',
                        help='the index of training and testing data partition set',
                        default='0', type=int)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.db == 'RAP':
        """Load RAP database"""
        from utils.rap_db import RAP
        db = RAP(os.path.join('data', args.db), args.par_set_id)
    else:
        """Load PETA dayanse"""
        from utils.peta_db import PETA
        db = PETA(os.path.join('data', args.db), args.par_set_id)

    f = open(args.pkl, 'rb')
    attr = cPickle.load(f)

    #print db.train_ind

    for i in xrange(1,5):
        print attr[i]  

    print db.evaluate_mA(attr, db.test_ind)
