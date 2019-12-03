#!/usr/bin/evn bash

set -e

export CUDA_VISIBLE_DEVICES=""

find ./experiments/best_params -type f -name "*.yml" | sort | while read fname
do
    for c in corrected original
    do
        BASE_NAME="semeval.abortion.${c}.supervised"
        python -m gcn.train \
            ./data/$BASE_NAME/$BASE_NAME \
            $fname --run-test
    done
done
