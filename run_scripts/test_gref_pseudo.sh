python test.py \
    --model lavt \
    --swin_type base \
    --dataset refcocog \
    --splitBy google \
    --split val \
    --resume ./checkpoints/model_best_gref_google.pth \
    --workers 4 \
    --ddp_trained_weights \
    --window12 \
    --ck_bert ./bert/models \
    --img_size 480 \
    --device cuda:1 \
    