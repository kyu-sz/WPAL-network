#!/usr/bin/env bash
cd `dirname "${BASH_SOURCE[0]}"`/../../..
./tools/estimate_param.py --setid 0 --net ./data/snapshots/GOOGLENET_SPP_RAP/0/RAP/googlenet_spp_rap_iter_130000.caffemodel --def ./models/GOOGLENET_SPP_RAP/test_net.prototxt --gpu 3 --db RAP
