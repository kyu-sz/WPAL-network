#!/usr/bin/env bash
cd `dirname "${BASH_SOURCE[0]}"`/../../..
./experiments/scripts/wpal_net.sh 2 GOOGLENET_PETA data/pretrained/bvlc_googlenet.caffemodel ProcessedPeta 0 --set TRAIN.BATCH_SIZE 48
