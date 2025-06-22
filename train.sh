#!/bin/bash
python train.py \
    --batch_size 1 \
    --epochs 100 \
    --lr 0.001\
    --lod 1 10 \
    --data_root "OSTFemur_Dataset" \
    --log_dir "Training" \
    --loss_type l2 \
    --optim adam \
    --lat_dims 64 \
    --deformer_nf 100 \
    --samples 10000 \
    --epsilon 0.5 \