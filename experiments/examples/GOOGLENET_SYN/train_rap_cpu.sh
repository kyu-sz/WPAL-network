#!/usr/bin/env bash
cd `dirname "${BASH_SOURCE[0]}"`/../../..
./experiments/scripts/wpal_net.sh -1 GOOGLENET_RAP_SYN ./data/pretrained/bvlc_googlenet.caffemodel RAP 0 --set TRAIN.BATCH_SIZE 16
