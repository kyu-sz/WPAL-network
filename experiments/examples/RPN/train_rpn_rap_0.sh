#!/usr/bin/env bash
cd `dirname "${BASH_SOURCE[0]}"`/../../..
./experiments/scripts/wpal_net.sh 0 RPN_RAP data/pretrained/FASTER_RCNN.caffemodel RAP 0
