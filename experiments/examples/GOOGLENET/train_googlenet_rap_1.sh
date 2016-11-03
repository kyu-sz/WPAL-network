#!/usr/bin/env bash
cd `dirname "${BASH_SOURCE[0]}"`/../../..
./experiments/scripts/wpal_net.sh 2 GOOGLENET_RAP data/pretrained/bvlc_googlenet.caffemodel RAP 1
