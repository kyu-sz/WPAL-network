#!/usr/bin/env bash
cd `dirname "${BASH_SOURCE[0]}"`/../../..
./experiments/scripts/wpal_net.sh 1 GOOGLENET_RAP ./data/snapshots/GOOGLENET_RAP/0/RAP/googlenet_rap_iter_20000.caffemodel RAP 0 --set TRAIN.BATCH_SIZE 24
