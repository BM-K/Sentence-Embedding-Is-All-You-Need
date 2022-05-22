# KoSentenceBERT
[[Github]](https://github.com/UKPLab/sentence-transformers) Official implementation of SBERT. <br>
Korean SentenceBERT : Korean Sentence Embeddings using Siamese BERT-Networks.

## Quick start
- If you want to do inference quickly, download the pre-trained models and then you can start some downstream tasks.
```
bash get_model_checkpoint.sh
cd KoSBERT/
python SemanticSearch.py
```

## Training
- Before training or evaluation, please download the datasets by running
    ```
    bash get_model_dataset.sh
    ```
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
  4. Epochs: 1(BERT), 2(RoBERTa)
  
- Continued Training STS
  1. Pooling Method: MEAN strategy
  2. Batch Size: 32
  3. Evaluation Steps: 1000
  4. Epochs: 4

### Semantic Search

```python
from sentence_transformers import SentenceTransformer, util
import numpy as np

model_path = '../Checkpoint/KoSBERT/kosbert-klue-bert-base'

embedder = SentenceTransformer(model_path)

# Corpus with example sentences
corpus = ['한 남자가 음식을 먹는다.',
          '한 남자가 빵 한 조각을 먹는다.',
          '그 여자가 아이를 돌본다.',
          '한 남자가 말을 탄다.',
          '한 여자가 바이올린을 연주한다.',
          '두 남자가 수레를 숲 솦으로 밀었다.',
          '한 남자가 담으로 싸인 땅에서 백마를 타고 있다.',
          '원숭이 한 마리가 드럼을 연주한다.',
          '치타 한 마리가 먹이 뒤에서 달리고 있다.']

corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

# Query sentences:
queries = ['한 남자가 파스타를 먹는다.',
           '고릴라 의상을 입은 누군가가 드럼을 연주하고 있다.',
           '치타가 들판을 가로 질러 먹이를 쫓는다.']

# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
top_k = 5
for query in queries:
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    cos_scores = cos_scores.cpu()

    #We use np.argpartition, to only partially sort the top_k results
    top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")

    for idx in top_results[0:top_k]:
        print(corpus[idx].strip(), "(Score: %.4f)" % (cos_scores[idx]))

```

- Results are as follows :

```

Query: 한 남자가 파스타를 먹는다.

Top 5 most similar sentences in corpus:
한 남자가 음식을 먹는다. (Score: 0.6141)
한 남자가 빵 한 조각을 먹는다. (Score: 0.5952)
한 남자가 말을 탄다. (Score: 0.1231)
한 남자가 담으로 싸인 땅에서 백마를 타고 있다. (Score: 0.0752)
두 남자가 수레를 숲 솦으로 밀었다. (Score: 0.0486)


======================


Query: 고릴라 의상을 입은 누군가가 드럼을 연주하고 있다.

Top 5 most similar sentences in corpus:
원숭이 한 마리가 드럼을 연주한다. (Score: 0.6656)
치타 한 마리가 먹이 뒤에서 달리고 있다. (Score: 0.2988)
한 여자가 바이올린을 연주한다. (Score: 0.1566)
한 남자가 말을 탄다. (Score: 0.1112)
한 남자가 담으로 싸인 땅에서 백마를 타고 있다. (Score: 0.0262)


======================


Query: 치타가 들판을 가로 질러 먹이를 쫓는다.

Top 5 most similar sentences in corpus:
치타 한 마리가 먹이 뒤에서 달리고 있다. (Score: 0.7570)
두 남자가 수레를 숲 솦으로 밀었다. (Score: 0.3658)
원숭이 한 마리가 드럼을 연주한다. (Score: 0.3583)
한 남자가 말을 탄다. (Score: 0.0505)
그 여자가 아이를 돌본다. (Score: -0.0087)
```

### Clustering 
```python
from sentence_transformers import SentenceTransformer, util
import numpy as np

model_path = '../Checkpoint/KoSBERT/kosbert-klue-bert-base'

embedder = SentenceTransformer(model_path)

# Corpus with example sentences
corpus = ['한 남자가 음식을 먹는다.',
          '한 남자가 빵 한 조각을 먹는다.',
          '그 여자가 아이를 돌본다.',
          '한 남자가 말을 탄다.',
          '한 여자가 바이올린을 연주한다.',
          '두 남자가 수레를 숲 솦으로 밀었다.',
          '한 남자가 담으로 싸인 땅에서 백마를 타고 있다.',
          '원숭이 한 마리가 드럼을 연주한다.',
          '치타 한 마리가 먹이 뒤에서 달리고 있다.',
          '한 남자가 파스타를 먹는다.',
          '고릴라 의상을 입은 누군가가 드럼을 연주하고 있다.',
          '치타가 들판을 가로 질러 먹이를 쫓는다.']

corpus_embeddings = embedder.encode(corpus)

# Then, we perform k-means clustering using sklearn:
from sklearn.cluster import KMeans

num_clusters = 5
clustering_model = KMeans(n_clusters=num_clusters)
clustering_model.fit(corpus_embeddings)
cluster_assignment = clustering_model.labels_

clustered_sentences = [[] for i in range(num_clusters)]
for sentence_id, cluster_id in enumerate(cluster_assignment):
    clustered_sentences[cluster_id].append(corpus[sentence_id])

for i, cluster in enumerate(clustered_sentences):
    print("Cluster ", i+1)
    print(cluster)
    print("")
```
- Results are as follows:
```
Cluster  1
['한 남자가 음식을 먹는다.', '한 남자가 빵 한 조각을 먹는다.', '한 남자가 파스타를 먹는다.']

Cluster  2
['원숭이 한 마리가 드럼을 연주한다.', '고릴라 의상을 입은 누군가가 드럼을 연주하고 있다.']

Cluster  3
['한 남자가 말을 탄다.', '두 남자가 수레를 숲 솦으로 밀었다.', '한 남자가 담으로 싸인 땅에서 백마를 타고 있다.']

Cluster  4
['치타 한 마리가 먹이 뒤에서 달리고 있다.', '치타가 들판을 가로 질러 먹이를 쫓는다.']

Cluster  5
['그 여자가 아이를 돌본다.', '한 여자가 바이올린을 연주한다.']
```
