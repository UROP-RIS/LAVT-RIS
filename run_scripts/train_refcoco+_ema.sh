#!/bin/bash
mkdir -p ./models/refcoco+

gpu="0,1,2"
export CUDA_VISIBLE_DEVICES=$gpu
np=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

torchrun \
    --nproc_per_node=$np \
    --master_port=12345 \
    train_ema.py \
    --model lavt \
    --dataset refcoco+ \
    --pseudo_dataset unc+ \
    --model_id refcoco+ \
    --batch-size 11 \
    --lr 0.00005 \
    --workers 12 \
    --wd 1e-2 \
    --swin_type base \
    --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth \
    --epochs 20 \
    --configs ./configs/ema_unc+_multi_text.json \
    --img_size 480 \
    --resume checkpoints/refcoco+_correction.pth \
    --pin_mem true \
    --ck_bert ./bert/models \
    2>&1 | tee ./models/refcoco+/output