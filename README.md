# Korean-Sentence-Embedding
🍭 Korean sentence embedding repository. You can download the pre-trained models and inference right away, also it provides environments where individuals can train models.

## Baseline Models
Baseline models used for korean sentence embedding - [KLUE-PLMs](https://github.com/KLUE-benchmark/KLUE/blob/main/README.md)

| Model                | Embedding size | Hidden size | # Layers | # Heads |
|----------------------|----------------|-------------|----------|---------|
| KLUE-BERT-base            | 768            | 768         | 12       | 12      |
| KLUE-RoBERTa-base         | 768            | 768         | 12       | 12      |

`NOTE`:  All the pretrained models are uploaded in Huggingface Model Hub. Check https://huggingface.co/klue.
<br>

## How to start
- Get datasets to train or test.
```
bash get_model_dataset.sh
```
- If you want to do inference quickly, download the pre-trained models and then you can start some downstream tasks.
```
bash get_model_checkpoint.sh
cd KoSBERT/
python SemanticSearch.py
```

## Available Models
1. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks [[SBERT]-[EMNLP 2019]](https://arxiv.org/abs/1908.10084)
2. SimCSE: Simple Contrastive Learning of Sentence Embeddings [[SimCSE]-[EMNLP 2021]](https://arxiv.org/abs/2104.08821)

## Datasets
- [kakao brain KorNLU Datasets](https://github.com/kakaobrain/KorNLUDatasets)

### KoSentenceBERT
- 🤗 [Model Training](https://github.com/BM-K/Sentence-Embedding-is-all-you-need/tree/main/KoSBERT)
- Dataset
    - Train: snli_1.0_train.ko.tsv (First phase, training NLI), sts-train.tsv (Second phase, continued training STS)
    - Valid: sts-dev.tsv
    - Test: sts-test.tsv

### KoSimCSE
- 🤗 [Model Training](https://github.com/BM-K/Sentence-Embedding-is-all-you-need/tree/main/KoSimCSE)
- Dataset
    - Train: snli_1.0_train.ko.tsv + multinli.train.ko.tsv
    - Valid: sts-dev.tsv
    - Test: sts-test.tsv

## Performance
- Semantic Textual Similarity test set results <br>

| Model                  | AVG | Cosine Pearson | Cosine Spearman | Euclidean Pearson | Euclidean Spearman | Manhattan Pearson | Manhattan Spearman | Dot Pearson | Dot Spearman |
|------------------------|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| KoSBERT<sup>†</sup><sub>SKT</sub>    | 77.40 | 78.81 | 78.47 | 77.68 | 77.78 | 77.71 | 77.83 | 75.75 | 75.22 |
| KoSBERT<sub>base</sub>               | 80.39 | 82.13 | 82.25 | 80.67 | 80.75 | 80.69 | 80.78 | 77.96 | 77.90 |
| KoSRoBERTa<sub>base</sub>            | 81.64 | 81.20 | 82.20 | 81.79 | 82.34 | 81.59 | 82.20 | 80.62 | 81.25 |
| | | | | | | | | |
| KoSimCSE-BERT<sup>†</sup><sub>SKT</sub>   | 81.32 | 82.12 | 82.56 | 81.84 | 81.63 | 81.99 | 81.74 | 79.55 | 79.19 |
| KoSimCSE-BERT<sub>base</sub>              | 81.56 | 83.05 | 83.33 | 82.62 | 82.96 | 82.78 | 83.09 | 77.97 | 76.70 |
| KoSimCSE-RoBERTa<sub>base</sub>           | 83.35 | 83.91 | 84.22 | 83.60 | 84.07 | 83.64 | 84.04 | 82.01 | 81.32 |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

- [KoSBERT<sup>†</sup><sub>SKT</sub>](https://github.com/BM-K/KoSentenceBERT-SKT)
- [KoSimCSE-BERT<sup>†</sup><sub>SKT</sub>](https://github.com/BM-K/KoSimCSE-SKT)

## Downstream Tasks
- KoSBERT: Semantic Search, Clustering
```
python SemanticSearch.py
python Clustering.py
```
- KoSimCSE: Semantic Search
```
python SemanticSearch.py
```
### Semantic Search (KoSBERT)
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

### Clustering (KoSBERT)
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

## References

```bibtex
@misc{park2021klue,
    title={KLUE: Korean Language Understanding Evaluation},
    author={Sungjoon Park and Jihyung Moon and Sungdong Kim and Won Ik Cho and Jiyoon Han and Jangwon Park and Chisung Song and Junseong Kim and Yongsook Song and Taehwan Oh and Joohong Lee and Juhyun Oh and Sungwon Lyu and Younghoon Jeong and Inkwon Lee and Sangwoo Seo and Dongjun Lee and Hyunwoo Kim and Myeonghwa Lee and Seongbo Jang and Seungwon Do and Sunkyoung Kim and Kyungtae Lim and Jongwon Lee and Kyumin Park and Jamin Shin and Seonghyun Kim and Lucy Park and Alice Oh and Jung-Woo Ha and Kyunghyun Cho},
    year={2021},
    eprint={2105.09680},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
@inproceedings{gao2021simcse,
   title={{SimCSE}: Simple Contrastive Learning of Sentence Embeddings},
   author={Gao, Tianyu and Yao, Xingcheng and Chen, Danqi},
   booktitle={Empirical Methods in Natural Language Processing (EMNLP)},
   year={2021}
}
@article{ham2020kornli,
  title={KorNLI and KorSTS: New Benchmark Datasets for Korean Natural Language Understanding},
  author={Ham, Jiyeon and Choe, Yo Joong and Park, Kyubyong and Choi, Ilji and Soh, Hyungjoon},
  journal={arXiv preprint arXiv:2004.03289},
  year={2020}
}
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "http://arxiv.org/abs/1908.10084",
}
```
## ToDo
- [ ] Huggingface model porting
