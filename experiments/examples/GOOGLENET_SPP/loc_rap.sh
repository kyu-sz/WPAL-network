#!/usr/bin/env bash
cd `dirname "${BASH_SOURCE[0]}"`/../../..
./tools/loc.py --setid 0 --net ./data/snapshots/GOOGLENET_SPP_RAP/0/googlenet_rap_352_best.caffemodel --def ./models/GOOGLENET_SPP_RAP/test_net.prototxt --gpu 1 --db RAP --detector-weight ./output/binding.pkl --set TEST.SCALE 352
