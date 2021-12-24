#!/bin/bash

echo "Start Training (BERT-BASE)"

CUDA_VISIBLE_DEVICES=0 python main.py \
  --model klue/bert-base \
  --test False \
  --max_len 50 \
  --batch_size 512 \
  --epochs 2 \
  --eval_steps 250 \
  --lr 0.0001 \
  --warmup_ratio 0.1 \
  --temperature 0.05 \
  --path_to_data ../Dataset/ \
  --train_data train_nli.tsv \
  --valid_data valid_sts.tsv

echo "Start Testing (BERT-BASE)"

CUDA_VISIBLE_DEVICES=0 python main.py \
  --model klue/bert-base \
  --train False \
  --test True \
  --max_len 50 \
  --batch_size 512 \
  --temperature 0.05 \
  --path_to_data ../Dataset/ \
  --test_data test_sts.tsv \
  --path_to_saved_model output/kosimcse-klue-bert-base.pt
  
 echo "Start Training (RoBERTa-BASE)"

CUDA_VISIBLE_DEVICES=0 python main.py \
  --model klue/roberta-base \
  --test False \
  --max_len 50 \
  --batch_size 512 \
  --epochs 2 \
  --eval_steps 125 \
  --lr 0.0001 \
  --warmup_ratio 0.2 \
  --temperature 0.05 \
  --path_to_data ../Dataset/ \
  --train_data train_nli.tsv \
  --valid_data valid_sts.tsv

echo "Start Testing (RoBERTa-BASE)"

CUDA_VISIBLE_DEVICES=0 python main.py \
  --model klue/roberta-base \
  --train False \
  --test True \
  --max_len 50 \
  --batch_size 512 \
  --temperature 0.05 \
  --path_to_data ../Dataset/ \
  --test_data test_sts.tsv \
  --path_to_saved_model output/kosimcse-klue-roberta-base.pt
