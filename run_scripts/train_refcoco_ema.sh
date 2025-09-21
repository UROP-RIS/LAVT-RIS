#!/bin/bash
mkdir -p ./models/refcoco

gpu="0,2,3,4"
export CUDA_VISIBLE_DEVICES=$gpu
np=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

torchrun \
    --nproc_per_node=$np \
    --master_port=12348 \
    train_ema.py \
    --model lavt \
    --dataset refcoco \
    --model_id refcoco \
    --pseudo_dataset unc \
    --batch-size 11 \
    --lr 0.00005 \
    --workers 12 \
    --wd 1e-2 \
    --swin_type base \
    --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth \
    --configs "configs/ema_unc_multi_text.json" \
    --epochs 20 \
    --img_size 480 \
    --pin_mem true \
    --ck_bert ./bert/models \
    --resume checkpoints/refcoco_correction.pth \
    2>&1 | tee ./models/refcoco/output.log