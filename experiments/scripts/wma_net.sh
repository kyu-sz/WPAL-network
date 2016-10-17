#!/bin/bash
# Usage:
# ./experiments/scripts/wma_net.sh GPU NET SNAPSHOT DB_SET [options args to {train,test}_net.py]
#
# Example:
# ./experiments/scripts/wma_net.sh 0 VGG_CNN_S pretrained 0 \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}
SNAPSHOT=$3
DB_SET=$4

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

ITERS=40000

LOG="experiments/logs/wna_net_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

if [ ${SNAPSHOT} = "pretrained" ]
then
    WEIGHTS=data/pretrained/${NET}.caffemodel 
else
    WEIGHTS=data/snapshots/${NET}/${DB_SET}/${SNAPSHOT}.caffemodel
fi

time ./tools/train_net.py --gpu ${GPU_ID} \
  --solver models/${NET}/solver.prototxt \
  --weights ${WEIGHTS} \
  --setid ${DB_SET} \
  --outputdir data/snapshots/${NET}/${DB_SET} \
  --iters ${ITERS} \
  ${EXTRA_ARGS}

set +x
NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
set -x

time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${NET}/test_net.prototxt \
  --setid ${DB_SET} \
  --net ${NET_FINAL} \
  ${EXTRA_ARGS}
