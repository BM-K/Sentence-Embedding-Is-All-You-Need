# KoSentenceT5
KoSentenceT5 : Korean Sentence Embeddings using T5. <br>
> **Warning** <br>
> This repository uses ETRI-T5 model and does not provide it. You can download T5 model from [here](https://aiopen.etri.re.kr/service_dataset.php).

## Training 
- Before training or evaluation, please download the datasets by running
```
bash get_model_dataset.sh
```
### Train KoSentenceT5
  ```
  python main.py \
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
  ```
### Evaluation
  ```
  python main.py \
    --model etri-t5 \
    --train False \
    --test True \
    --max_len 110 \
    --batch_size 64 \
    --temperature 0.05 \
    --path_to_data ../Dataset/ \
    --test_data test_sts.tsv \
  ```

### Run Examples
```
bash run_example.sh
```
