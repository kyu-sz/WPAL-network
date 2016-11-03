#!/usr/bin/env bash
cd `dirname "${BASH_SOURCE[0]}"`/../../..
./experiments/scripts/wpal_net.sh -1 RPN_RAP data/pretrained/FASTER_RCNN.caffemodel RAP 0 --set TRAIN.BATCH_SIZE 16
