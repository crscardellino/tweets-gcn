#!/usr/bin/evn bash

set -e

export CUDA_VISIBLE_DEVICES=""

find ./experiments/grid_params -type f -name "*.yml" | sort | while read fname
do
    BASE_NAME="semeval.abortion.original.semisupervised"
    python -m gcn.train \
        ./data/$BASE_NAME/$BASE_NAME.0 \
        $fname
done
