#!/usr/bin/env bash
cd `dirname "${BASH_SOURCE[0]}"`/../..
./experiments/scripts/wma_net.sh 1 VGG_S_MLL data/snapshots/VGG_S_MLL/0/vgg_s_mll_iter_0.caffemodel RAP 0
