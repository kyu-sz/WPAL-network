#!/usr/bin/env bash
cd `dirname "${BASH_SOURCE[0]}"`/../../..
./tools/test_net.py --setid 0 --net ./data/snapshots/GOOGLENET_SPP_PETA/0/ProcessedPeta/googlenet_spp_peta_iter_75000.caffemodel --def ./models/GOOGLENET_SPP_PETA/test_net.prototxt --gpu 0 --db ProcessedPeta
