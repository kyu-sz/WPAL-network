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

"""WPAL-net cfg system.
This file specifies default cfg options for WPAL-net. You should not
change values in this file. Instead, you should write a cfg file (in yaml)
and use cfg_from_file(yaml_file) to load it and override the default options.
Most tools in $ROOT/tools take a --cfg option to specify an override file.
    - See tools/{train,test}_net.py for example code that uses cfg_from_file()
    - See experiments/cfgs/*.yml for example YAML cfg override files
"""

import os
import os.path as osp

import numpy as np
from easydict import EasyDict as edict

__C = edict()
# Get cfg by:
#   from wpal_net.config import cfg
cfg = __C

#
# Training options
#

__C.TRAIN = edict()

# Scales to use during training (can list multiple scales)
# Each scale is the pixel size of an image's longer sides
__C.TRAIN.SCALES = (368, 384, 400, 416, 432, 448, 464, 480, 496)

# Max pixel area size of a scaled input image
__C.TRAIN.MAX_AREA = 96768

# Minibatch size (number of samples per round)
__C.TRAIN.BATCH_SIZE = 16

# Use horizontally-flipped images during training?
__C.TRAIN.USE_FLIPPED = True

# Iterations between tests
__C.TRAIN.TEST_ITERS = 10000000

# Iterations between snapshots
__C.TRAIN.SNAPSHOT_ITERS = 5000

# solver.prototxt specifies the snapshot path prefix, this adds an optional
# infix to yield the path: <prefix>[_<infix>]_iters_XYZ.caffemodel
__C.TRAIN.SNAPSHOT_INFIX = ''

# Make minibatches from images that have similar aspect ratios (i.e. both
# tall and thin or both short and wide) in order to avoid wasting computation
# on zero-padding.
__C.TRAIN.ASPECT_GROUPING = True

# Whether to do flipping during training
__C.TRAIN.DO_FLIP = True

# Value range of RGB Jittering.
__C.TRAIN.RGB_JIT = 16

# Number of detectors to reserve when finding unutilized detectors.
__C.TRAIN.NUM_RESERVE_DETECTOR = 64

#
# Testing options
#

__C.TEST = edict()

# Scale to use during testing
# The scale is the pixel size of a test image's longer sides
__C.TEST.SCALE = 432

# Max pixel area size of a scaled input image
__C.TEST.MAX_AREA = 96768


#
# MISC
#

# Min pixel size of an input image's shorter side.
__C.MIN_SIZE = 96

# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

# For reproducibility
__C.RNG_SEED = 3

# A small number that's used many times
__C.EPS = 1e-14

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

# Data directory
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))

# Model directory
__C.MODELS_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'models', 'vgg_cnn_f'))

# Place outputs under an experiments directory
__C.EXP_DIR = 'default'

# Default GPU device id
__C.GPU_ID = 0

__C.NUM_DETECTOR = 2048


def get_output_dir(imdb, net=None):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'output', __C.EXP_DIR, imdb.name))
    if net is not None:
        outdir = osp.join(outdir, net.name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


def _merge_a_into_b(a, b):
    """Merge cfg dictionary a into cfg dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if not b.has_key(k):
            raise KeyError('{} is not a valid cfg key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for cfg key: {}').format(type(b[k]),
                                                            type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under cfg key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a cfg file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
    """Set cfg keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert d.has_key(subkey)
            d = d[subkey]
        subkey = key_list[-1]
        assert d.has_key(subkey)
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value
