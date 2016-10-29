#!/usr/bin/env bash
cd `dirname "${BASH_SOURCE[0]}"`/../../..
./experiments/scripts/wpal_net.sh 1 RPN_PETA data/pretrained/faster-rcnn.caffemodel ProcessedPeta 0
