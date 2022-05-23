#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 python main.py \
  --model etri-t5 \
  --multi_gpu True \
  --test False \
  --max_len 110 \
  --batch_size 64 \
  --epochs 2 \
  --eval_steps 125 \
  --lr 0.0001 \
  --warmup_ratio 0.01 \
  --temperature 0.05 \
  --path_to_data ../Dataset/ \
  --train_data train_nli.tsv \
  --valid_data valid_sts.tsv

CUDA_VISIBLE_DEVICES=1 python main.py \
  --model etri-t5 \
  --train False \
  --test True \
  --max_len 110 \
  --batch_size 64 \
  --temperature 0.05 \
  --path_to_data ../Dataset/ \
  --test_data test_sts.tsv \
  --path_to_saved_model output/
