#!/usr/bin/env bash
cd `dirname "${BASH_SOURCE[0]}"`/../../..
./experiments/scripts/wpal_net.sh 1 VGG_S_MLL_PETA data/pretrained/VGG_CNN_S.caffemodel ProcessedPeta 0
