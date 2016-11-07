#!/usr/bin/env bash
cd `dirname "${BASH_SOURCE[0]}"`/../../..
./tools/test_net.py --setid 0 --net ./data/snapshots/GOOGLENET_RAP/0/RAP/googlenet_rap_iter_130000.caffemodel --def ./models/GOOGLENET_RAP/test_net.prototxt --gpu 0 --db RAP

