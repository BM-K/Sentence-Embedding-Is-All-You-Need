#!/bin/bash

# bert-base
echo "First Step Training NLI Dataset (BERT-BASE)"
CUDA_VISIBLE_DEVICES=0 python training_nli.py --model klue/bert-base --batch 32 --evaluation_steps 1000 --epochs 1
echo "Second Step Continuously Training STS Dataset (BERT-BASE)"
CUDA_VISIBLE_DEVICES=0 python con_training_sts.py --model klue/bert-base --batch 32 --evaluation_steps 1000 --epochs 4

# roberta-base
echo "First Step Training NLI Dataset (ROBERTA-BASE)"
CUDA_VISIBLE_DEVICES=0 python training_nli.py --model klue/roberta-base --batch 32 --evaluation_steps 1000 --epochs 1
echo "Second Step Continuously Training STS Dataset (ROBERTA-BASE)"
CUDA_VISIBLE_DEVICES=0 python con_training_sts.py --model klue/roberta-base --batch 32 --evaluation_steps 1000 --epochs 4

# roberta-large
echo "First Step Training NLI Dataset (ROBERAT-LARGE)"
CUDA_VISIBLE_DEVICES=0 python training_nli.py --model klue/roberta-large --batch 32 --evaluation_steps 1000 --epochs 1
echo "Second Step Continuously Training STS Dataset (ROBERTA-LARGE)"
CUDA_VISIBLE_DEVICES=0 python con_training_sts.py --model klue/roberta-large --batch 32 --evaluation_steps 1000 --epochs 4

