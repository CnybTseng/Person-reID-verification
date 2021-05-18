#!/bin/sh

python test.py \
    --gpu_ids 0 \
    --name $1 \
    --test_dir /data/tseng/dataset/reid/market1501/Market-1501-v15.09.15/pytorch/ \
    --which_epoch 59