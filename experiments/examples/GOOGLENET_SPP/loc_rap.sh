#!/usr/bin/env bash
cd `dirname "${BASH_SOURCE[0]}"`/../../..
./tools/loc.py --setid 0 --net ./data/snapshots/GOOGLENET_SPP_RAP/0/googlenet_spp_rap_448_best.caffemodel --def ./models/GOOGLENET_SPP_RAP/test_net.prototxt --db RAP --detector-weight ./output/rap_448_detector.pkl --attr-id -2 --cfg experiments/cfgs/spp.yml --display 0 --max-count 5
