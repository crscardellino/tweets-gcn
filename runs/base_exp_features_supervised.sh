#!/usr/bin/evn bash

set -e

(for c in corrected original
do
    BASE_NAME="semeval.abortion.${c}.supervised"
    python data_processing.py \
        --char-ngrams 2 3 4 \
        --graph-hashtags \
        --graph-mentions \
        --graph-ngrams 3 4 5 \
        --ignore-hashtag semst \
        --min-docs 2 \
        --remove-links \
        --remove-mentions \
        --remove-numeric \
        --reduce-tweet-word-len \
        --tweet-lowercase \
        ./data/$BASE_NAME/$BASE_NAME.csv.gz \
        ./data/$BASE_NAME/$BASE_NAME &
done
wait)
