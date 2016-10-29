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


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='train Weakly-supervised Pedestrian Attribute Localization Network')
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
    print db.evaluate_example_based(attr, db.test_ind)
