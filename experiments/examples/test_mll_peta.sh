#!/usr/bin/env bash
./tools/test_net.py --setid 0 --net ./data/snapshots/VGG_S_MLL_PETA/0/vgg_s_mll_peta_iter_28000.caffemodel --def ./models/VGG_S_MLL_PETA/test_net.prototxt --gpu 2 --db ProcessedPeta
