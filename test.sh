#!/bin/bash
python test.py \
    --batch_size 1 \
    --epochs 100 \
    --lr 0.001\
    --lod 1 10 \
    --data_root "OSTFemur_Dataset" \
    --log_dir "Evaluation" \
    --loss_type l2 \
    --optim adam \
    --lat_dims 64 \
    --deformer_nf 100 \
    --samples 10000 \
    --epsilon 0.5 \
    --resume "split_data_new_03_05_fulldata_logs_new/checkpoint_latest_deformer_193.pth.tar"