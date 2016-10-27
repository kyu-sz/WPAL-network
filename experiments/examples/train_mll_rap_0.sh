#!/usr/bin/env bash
cd `dirname "${BASH_SOURCE[0]}"`/../..
./experiments/scripts/wma_net.sh 0 VGG_S_MLL_RAP data/snapshots/VGG_S_MLL_RAP/0/vgg_s_mll_rap_iter_35000.caffemodel RAP 0
