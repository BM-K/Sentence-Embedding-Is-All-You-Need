# KoSimCSE
KoSimCSE : Sentence Embeddings using contrastive learning

## Training
- Before training or evaluation, please download the datasets by running
```
bash get_model_dataset.sh
```
### Train KoSimCSE
  ```
  python main.py \
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
  ```
### Evaluation
  ```
  python main.py \
    --model klue/bert-base \
    --train False \
    --test True \
    --max_len 50 \
    --batch_size 512 \
    --temperature 0.05 \
    --path_to_data ../Dataset/ \
    --test_data test_sts.tsv \
    --path_to_saved_model output/kosimcse-klue-bert-base.pt
  ```

### Run Examples
```
bash run_example.sh
```
### Hyperparameters
- Train KoSimCSE (BERT BASE)
  1. Pooling Method: CLS strategy
  2. Batch Size: 512
  3. Evaluation Steps: 250
  4. Epochs: 2
  5. Token Max Length: 50
  6. Learning Rate: 0.0001
  7. Warmup Ratio: 0.1
  8. Temperature: 0.05
  
- Train KoSimCSE (RoBERTa BASE)
  1. Pooling Method: CLS strategy
  2. Batch Size: 512
  3. Evaluation Steps: 125
  4. Epochs: 2
  5. Token Max Length: 50
  6. Learning Rate: 0.0001
  7. Warmup Ratio: 0.2
  8. Temperature: 0.05
