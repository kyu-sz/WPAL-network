#!/usr/bin/env bash
cd `dirname "${BASH_SOURCE[0]}"`/../../..
./tools/test_net.py --setid 0 --net ./data/snapshots/RPN_RAP/0/RAP/rpn_rap_iter_50000.caffemodel --def ./models/RPN_RAP/test_net.prototxt --gpu 2 --db RAP
