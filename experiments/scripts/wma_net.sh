#!/bin/bash
# Usage:
# ./experiments/scripts/wma_net.sh GPU NET [options args to {train,test}_net.py]
#
# Example:
# ./experiments/scripts/wma_net.sh 0 VGG_CNN_S \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

ITERS=40000

LOG="experiments/logs/wna_net_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu ${GPU_ID} \
  --solver models/${NET}/solver.prototxt \
  --weights data/pretrained_models/${NET}.caffemodel \
  --iters ${ITERS} \
  ${EXTRA_ARGS}

set +x
NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
set -x

time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${NET}/test.prototxt \
  --net ${NET_FINAL} \
  ${EXTRA_ARGS}