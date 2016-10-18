#!/usr/bin/env bash
cd `dirname "${BASH_SOURCE[0]}"`/../..
./experiments/scripts/wma_net.sh 1 VGG_S_MLL data/pretrained/VGG_CNN_S.caffemodel RAP 0
