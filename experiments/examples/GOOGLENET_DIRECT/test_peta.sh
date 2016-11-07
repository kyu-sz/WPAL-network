#!/usr/bin/env bash
cd `dirname "${BASH_SOURCE[0]}"`/../../..
./tools/test_net.py --setid 0 --net ./data/snapshots/GOOGLENET_RAP_DIRECT/0/ProcessedPeta/googlenet_peta_iter_10000.caffemodel --def ./models/GOOGLENET_PETA/test_net.prototxt --gpu 3 --db ProcessedPeta
