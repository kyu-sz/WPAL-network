#!/usr/bin/env bash
cd `dirname "${BASH_SOURCE[0]}"`/../../..
./experiments/scripts/wpal_net.sh 1 VGG_S_MLL_RAP ./data/pretrained/VGG_CNN_S.caffemodel RAP 0 --set TRAIN.BATCH_SIZE 64

