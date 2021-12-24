# KoSentenceBERT
Korean SentenceBERT : Sentence Embeddings using Siamese BERT-Networks using KLUE KoBERT and kakaobrain KorNLU dataset

## Training
- Two stage training
    - First step, training NLI dataset
    ```
    python training_nli.py --model klue/bert-base --batch 32 --evaluation_steps 1000 --epochs 1
    ```
    - Second step, continued training STS dataset
    ```
    python con_training_sts.py --model klue/bert-base --batch 32 --evaluation_steps 1000 --epochs 4
    ```
- Run Examples
  ```
  bash run_example.sh
  ```
### Hyperparameters
- Training NLI
  1. Pooling Method: MEAN strategy
  2. Batch Size: 32
  3. Evaluation Steps: 1000
  4. Epochs: 1
  
- Continued Training STS
  1. Pooling Method: MEAN strategy
  2. Batch Size: 32
  3. Evaluation Steps: 1000
  4. Epochs: 4
