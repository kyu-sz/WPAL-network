#!/usr/bin/env bash
cd `dirname "${BASH_SOURCE[0]}"`/../../..
./experiments/scripts/wpal_net.sh 2 GOOGLENET_PETA ./data/snapshots/GOOGLENET_PETA/0/ProcessedPeta/googlenet_peta_iter_20000.caffemodel ProcessedPeta 0 --set TRAIN.BATCH_SIZE 24
