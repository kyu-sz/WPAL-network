#!/usr/bin/env bash
cd `dirname "${BASH_SOURCE[0]}"`/../../..
./experiments/scripts/wpal_net.sh 2 GOOGLENET_RAP_SYN ./data/pretrained/bvlc_googlenet.caffemodel RAP 1 --set TRAIN.BATCH_SIZE 32
