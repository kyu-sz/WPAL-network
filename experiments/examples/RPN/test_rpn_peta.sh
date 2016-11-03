#!/usr/bin/env bash
cd `dirname "${BASH_SOURCE[0]}"`/../../..
./tools/test_net.py --setid 0 --net ./data/snapshots/RPN_PETA/0/ProcessedPeta/rpn_peta_iter_15000.caffemodel --def ./models/RPN_PETA/test_net.prototxt --gpu 3 --db ProcessedPeta
