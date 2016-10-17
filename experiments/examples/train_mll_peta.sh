#!/usr/bin/env bash
cd `dirname "${BASH_SOURCE[0]}"`/../..
./experiments/scripts/wma_net.sh 0 VGG_S_MLL_PETA data/pretrained/VGG_CNN_S.caffemodel ProcessedPeta 0
