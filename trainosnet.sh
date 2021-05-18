#!/bin/sh

python train.py \
    --gpu_ids 0 \
    --name ft_OSNet \
    --train_all \
    --batchsize 32  \
    --data_dir /data/tseng/dataset/reid/market1501/Market-1501-v15.09.15/pytorch/ \
    --erasing_p 0.5 \
    --use_osnet