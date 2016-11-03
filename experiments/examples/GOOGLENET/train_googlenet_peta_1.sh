#!/usr/bin/env bash
cd `dirname "${BASH_SOURCE[0]}"`/../../..
./experiments/scripts/wpal_net.sh 3 GOOGLENET_PETA data/pretrained/bvlc_googlenet.caffemodel ProcessedPeta 1
