#!/usr/bin/env bash
cd `dirname "${BASH_SOURCE[0]}"`/../../..
./tools/test_net.py --setid 0 --net ./data/snapshots/VGG_S_MLL_PETA/0/ProcessedPeta/vgg_s_mll_peta_iter_15000.caffemodel --def ./models/VGG_S_MLL_PETA/test_net.prototxt --gpu 3 --db ProcessedPeta
