#!/usr/bin/env bash
cd `dirname "${BASH_SOURCE[0]}"`/../../..
./experiments/scripts/wpal_net.sh 1 GOOGLENET_SPP_PETA ./data/snapshots/GOOGLENET_SPP_PETA/0/ProcessedPeta/googlenet_spp_peta_iter_5000.caffemodel ProcessedPeta 0 --set TRAIN.BATCH_SIZE 24
