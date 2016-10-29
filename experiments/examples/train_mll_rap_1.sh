#!/usr/bin/env bash
cd `dirname "${BASH_SOURCE[0]}"`/../..
./experiments/scripts/wpal_net.sh 2 VGG_S_MLL data/snapshots/VGG_S_MLL/1/vgg_s_mll_iter_32000.caffemodel RAP 1
