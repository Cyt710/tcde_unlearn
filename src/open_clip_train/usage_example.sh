#!/bin/bash

#Example of code execution on the CIFAR dataset

export PYTHONPATH =
python main.py \
    --dataset-type "cifar" \
    --epochs 10 \
    --batch-size 64 \
    --lr  1e-6\
    --wd  0.01\
    --model "ViT-B-32" \
    --pretrained "" \
    --name "clip" \
    --workers 8 \
    --precision amp \
    --log-every-n-steps 10 \
    --save-frequency 5  \
    --zeroshot-frequency 1 \
    --val-frequency 1 \
    --accum-freq 1 \
    --cliperase \
    --lambda-rm 1.0 \
    --lambda-fm 0.0 \
    --lambda-cm 3.0 \
    --lambda-unif 1.0\
    --lambda-dist 1.0\
    --lambda-ga 0.0\
    --lambda-gd 0.0\
    --lambda-kl 0.0\
    --cliperase-temperature 0.07 \
    --eval-forget-set \
    --save-forget-metrics \
    --val-split 0.3 \
    --logs "" \
    --device "cuda" \
    --forget-rule "keyword_based"\
    --forget-keywords "" \
    --aug-cfg use_timm=True re_prob=0.25 color_jitter=0.4\
    --seed 42
