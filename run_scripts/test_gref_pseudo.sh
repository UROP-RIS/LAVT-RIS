python test.py \
    --model lavt \
    --swin_type base \
    --dataset refcocog \
    --splitBy google \
    --split val \
    --resume ./checkpoints/gref_consistent_with_unc.pth  \
    --workers 4 \
    --ddp_trained_weights \
    --window12 \
    --ck_bert ./bert/models \
    --img_size 480 \
    --device cuda:1 \
    