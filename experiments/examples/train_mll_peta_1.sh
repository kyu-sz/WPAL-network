#!/usr/bin/env bash
cd `dirname "${BASH_SOURCE[0]}"`/../..
./experiments/scripts/wpal_net.sh 3 VGG_S_MLL_PETA data/snapshots/VGG_S_MLL_PETA/1/vgg_s_mll_peta_iter_23000.caffemodel ProcessedPeta 1
