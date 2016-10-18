#!/usr/bin/env bash
./tools/test_net.py --setid 0 --net ./data/snapshots/VGG_S_MLL/0/vgg_s_mll_iter_37000.caffemodel --def ./models/VGG_S_MLL/test_net.prototxt --gpu 2 --db RAP
