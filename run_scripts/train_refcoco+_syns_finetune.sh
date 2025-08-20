#!/bin/bash
mkdir -p ./models/refcoco+

gpu="4,5,6,7"
export CUDA_VISIBLE_DEVICES=$gpu
np=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

torchrun \
    --nproc_per_node=$np \
    --master_port=12346 \
    train_pseudo.py \
    --model lavt \
    --dataset refcoco+ \
    --pseudo_dataset unc unc+ \
    --model_id refcoco+ \
    --batch-size 12 \
    --lr 0.00005 \
    --workers 12 \
    --wd 1e-2 \
    --swin_type base \
    --configs "configs/main_pure.json" \
    --resume "./checkpoints/refcoco+_pseudo_consistent.pth" \
    --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth \
    --epochs 30 \
    --img_size 480 \
    --pin_mem true \
    --ck_bert ./bert/models \
    --output_dir "./checkpoints/ft_ralative_only_no_consistent"
    2>&1 | tee ./models/refcoco/output