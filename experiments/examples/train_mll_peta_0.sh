#!/usr/bin/env bash
cd `dirname "${BASH_SOURCE[0]}"`/../..
./experiments/scripts/wma_net.sh 1 VGG_S_MLL_PETA data/snapshots/VGG_S_MLL_PETA/0/vgg_s_mll_peta_iter_36000.caffemodel ProcessedPeta 0
