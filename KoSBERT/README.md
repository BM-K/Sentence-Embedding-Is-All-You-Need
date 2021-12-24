# Ko-Sentence-BERT
🌷 Korean SentenceBERT : Sentence Embeddings using Siamese BERT-Networks using ETRI KoBERT and kakaobrain KorNLU dataset

## Installation
 - ETRI KorBERT는 transformers 2.4.1 ~ 2.8.0에서만 동작하고 Sentence-BERT는 3.1.0 버전 이상에서 동작하여 라이브러리를 수정하였습니다. <br>
 - **huggingface transformer, sentence transformers, tokenizers** 라이브러리 코드를 직접 수정하므로 가상환경 사용을 권장합니다.  <br>
 - 사용한 Docker image는 Docker Hub에 첨부합니다. <br>
     - https://hub.docker.com/r/klbm126/kosbert_image/tags <br>
 - ETRI KoBERT를 사용하여 학습하였고 본 레파지토리에선 ETRI KoBERT를 제공하지 않습니다. <br>
 - **SKT KoBERT**를 사용한 버전은 다음 레파지토리에 공개되어 있습니다. <br>
     - https://github.com/BM-K/KoSentenceBERT_SKTBERT <br>
```
git clone https://github.com/BM-K/KoSentenceBERT.git
python -m venv .KoSBERT
. .KoSBERT/bin/activate
pip install -r requirements.txt
```
 - transformer, tokenizers, sentence_transformers 디렉토리를 .KoSBERT/lib/python3.7/site-packages/ 로 이동합니다. <br>
 - ETRI_KoBERT 모델과 tokenizer가 KoSentenceBERT 디렉토리 안에 존재하여야 합니다.<br>
 - ETRI 모델과 tokenizer는 다음 예시와 같이 불러옵니다 :

 ```python
 from ETRI_tok.tokenization_etri_eojeol import BertTokenizer
 self.auto_model = BertModel.from_pretrained('./ETRI_KoBERT/003_bert_eojeol_pytorch') 
 self.tokenizer = BertTokenizer.from_pretrained('./ETRI_KoBERT/003_bert_eojeol_pytorch/vocab.txt', do_lower_case=False)
 ```

## Train Models
 - 모델 학습을 원하시면 KoSentenceBERT 디렉토리 안에 KorNLUDatasets이 존재하여야 합니다. <br>
 - STS 학습 시 모델 구조에 맞게 데이터를 수정하여 사용하였으며, 데이터와 학습 방법은 아래와 같습니다 : <br><br>
KoSentenceBERT/KorNLUDatasets/KorSTS/tune_test.tsv <br>
<img src="https://user-images.githubusercontent.com/55969260/93304207-97afec00-f837-11ea-88a2-7256f2f1664e.png"></img><br>
*STS test 데이터셋의 일부* <br>
```
python training_nli.py      # NLI 데이터로만 학습
python training_sts.py      # STS 데이터로만 학습
python con_training_sts.py  # NLI 데이터로 학습 후 STS 데이터로 Fine-Tuning
```

## Pre-Trained Models
**pooling mode**는 **MEAN-strategy**를 사용하였으며, 학습시 모델은 output 디렉토리에 저장 됩니다. <br>
|디렉토리|학습방법|
|-----------|:----:|
|training_**nli**_ETRI_KoBERT-003_bert_eojeol|Only Train NLI|
|training_**sts**_ETRI_KoBERT-003_bert_eojeol|Only Train STS|
|training_**nli_sts**_ETRI_KoBERT-003_bert_eojeol|STS + NLI|

## Performance
Seed 고정, test set
|Model|Cosine Pearson|Cosine Spearman|Euclidean Pearson|Euclidean Spearman|Manhattan Pearson|Manhattan Spearman|Dot Pearson|Dot Spearman|
|:------------------------:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|NLl|67.96|70.45|71.06|70.48|71.17|70.51|64.87|63.04|
|STS|**80.43**|79.99|78.18|78.03|78.13|77.99|73.73|73.40|
|STS + NLI|80.10|**80.42**|**79.14**|**79.28**|**79.08**|**79.22**|**74.46**|**74.16**|

## Application Examples
- 생성 된 문장 임베딩을 다운 스트림 애플리케이션에 사용할 수 있는 방법에 대한 몇 가지 예를 제시합니다.
- STS + NLI pretrained 모델을 통해 진행합니다.

### Semantic Search
SemanticSearch.py는 주어진 문장과 유사한 문장을 찾는 작업입니다.<br>
먼저 Corpus의 모든 문장에 대한 임베딩을 생성합니다.
```python
from sentence_transformers import SentenceTransformer, util
import numpy as np

model_path = './output/training_nli_sts_ETRI_KoBERT-003_bert_eojeol'

embedder = SentenceTransformer(model_path)

# Corpus with example sentences
corpus = ['한 남자가 음식을 먹는다.',
          '한 남자가 빵 한 조각을 먹는다.',
          '그 여자가 아이를 돌본다.',
          '한 남자가 말을 탄다.',
          '한 여자가 바이올린을 연주한다.',
          '두 남자가 수레를 숲 속으로 밀었다.',
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
<br> 결과는 다음과 같습니다 :
```
========================


Query: 한 남자가 파스타를 먹는다.

Top 5 most similar sentences in corpus:
한 남자가 음식을 먹는다. (Score: 0.7557)
한 남자가 빵 한 조각을 먹는다. (Score: 0.6464)
한 남자가 담으로 싸인 땅에서 백마를 타고 있다. (Score: 0.2565)
한 남자가 말을 탄다. (Score: 0.2333)
두 남자가 수레를 숲 속으로 밀었다. (Score: 0.1792)


========================


Query: 고릴라 의상을 입은 누군가가 드럼을 연주하고 있다.

Top 5 most similar sentences in corpus:
원숭이 한 마리가 드럼을 연주한다. (Score: 0.6732)
치타 한 마리가 먹이 뒤에서 달리고 있다. (Score: 0.3401)
두 남자가 수레를 숲 속으로 밀었다. (Score: 0.1037)
한 남자가 음식을 먹는다. (Score: 0.0617)
그 여자가 아이를 돌본다. (Score: 0.0466)


=======================


Query: 치타가 들판을 가로 질러 먹이를 쫓는다.

Top 5 most similar sentences in corpus:
치타 한 마리가 먹이 뒤에서 달리고 있다. (Score: 0.7164)
두 남자가 수레를 숲 속으로 밀었다. (Score: 0.3216)
원숭이 한 마리가 드럼을 연주한다. (Score: 0.2071)
한 남자가 빵 한 조각을 먹는다. (Score: 0.1089)
한 남자가 음식을 먹는다. (Score: 0.0724)
```
### Clustering
Clustering.py는 문장 임베딩 유사성을 기반으로 유사한 문장을 클러스터링하는 예를 보여줍니다. <br>
이전과 마찬가지로 먼저 각 문장에 대한 임베딩을 계산합니다. <br>
```python
from sentence_transformers import SentenceTransformer, util
import numpy as np

model_path = './output/training_nli_sts_ETRI_KoBERT-003_bert_eojeol'

embedder = SentenceTransformer(model_path)

# Corpus with example sentences
corpus = ['한 남자가 음식을 먹는다.',
          '한 남자가 빵 한 조각을 먹는다.',
          '그 여자가 아이를 돌본다.',
          '한 남자가 말을 탄다.',
          '한 여자가 바이올린을 연주한다.',
          '두 남자가 수레를 숲 속으로 밀었다.',
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
결과는 다음과 같습니다 :
 ```
 Cluster  1
['두 남자가 수레를 숲 속으로 밀었다.', '치타 한 마리가 먹이 뒤에서 달리고 있다.', '치타가 들판을 가로 질러 먹이를 쫓는다.']

Cluster  2
['한 남자가 말을 탄다.', '한 남자가 담으로 싸인 땅에서 백마를 타고 있다.']

Cluster  3
['한 남자가 음식을 먹는다.', '한 남자가 빵 한 조각을 먹는다.', '한 남자가 파스타를 먹는다.']

Cluster  4
['그 여자가 아이를 돌본다.', '한 여자가 바이올린을 연주한다.']

Cluster  5
['원숭이 한 마리가 드럼을 연주한다.', '고릴라 의상을 입은 누군가가 드럼을 연주하고 있다.']
```
<img src = "https://user-images.githubusercontent.com/55969260/96250228-78001500-0fe9-11eb-9ee5-914705182a55.png">

## Downstream Tasks Demo
<img src = "https://user-images.githubusercontent.com/55969260/99897723-73610780-2cdf-11eb-9b71-3d31a309a53a.gif"> <br>
<img src = "https://user-images.githubusercontent.com/55969260/99897743-94295d00-2cdf-11eb-9b16-d6fec66e43d0.gif"> <br>
<img src = "https://user-images.githubusercontent.com/55969260/99897764-a73c2d00-2cdf-11eb-8bcf-0235d1fda0f7.gif"> <br>

## Citing
### KorNLU Datasets
```bibtex
@article{ham2020kornli,
  title={KorNLI and KorSTS: New Benchmark Datasets for Korean Natural Language Understanding},
  author={Ham, Jiyeon and Choe, Yo Joong and Park, Kyubyong and Choi, Ilji and Soh, Hyungjoon},
  journal={arXiv preprint arXiv:2004.03289},
  year={2020}
}
```
### Sentence Transformers: Multilingual Sentence Embeddings using BERT / RoBERTa / XLM-RoBERTa & Co. with PyTorch
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "http://arxiv.org/abs/1908.10084",
}

@article{reimers-2020-multilingual-sentence-bert,
    title = "Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation",
    author = "Reimers, Nils and Gurevych, Iryna",
    journal= "arXiv preprint arXiv:2004.09813",
    month = "04",
    year = "2020",
    url = "http://arxiv.org/abs/2004.09813",
}
```

