#!/usr/bin/env bash
cd `dirname "${BASH_SOURCE[0]}"`/../../..
./experiments/scripts/wpal_net.sh 2 GOOGLENET_SPP_RAP ./data/snapshots/GOOGLENET_SPP_RAP/0/RAP/googlenet_spp_rap_iter_60000.caffemodel RAP 0 --set TRAIN.BATCH_SIZE 16
