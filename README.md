# Korean-Sentence-Embedding
🍭 Korean sentence embedding repository. You can download the pre-trained models and inference right away, also it provides environments where individuals can train models.

## Quick tour
> **Note** <br>
> All the pretrained models are uploaded in Huggingface Model Hub. Check https://huggingface.co/BM-K
```python
import torch
from transformers import AutoModel, AutoTokenizer

def cal_score(a, b):
    if len(a.shape) == 1: a = a.unsqueeze(0)
    if len(b.shape) == 1: b = b.unsqueeze(0)

    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    return torch.mm(a_norm, b_norm.transpose(0, 1)) * 100

model = AutoModel.from_pretrained('BM-K/KoSimCSE-roberta-multitask')  # or 'BM-K/KoSimCSE-bert-multitask'
tokenizer = AutoTokenizer.from_pretrained('BM-K/KoSimCSE-roberta-multitask')  # or 'BM-K/KoSimCSE-bert-multitask'

sentences = ['치타가 들판을 가로 질러 먹이를 쫓는다.',
             '치타 한 마리가 먹이 뒤에서 달리고 있다.',
             '원숭이 한 마리가 드럼을 연주한다.']

inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
embeddings, _ = model(**inputs, return_dict=False)

score01 = cal_score(embeddings[0][0], embeddings[1][0])  # 84.09
# '치타가 들판을 가로 질러 먹이를 쫓는다.' @ '치타 한 마리가 먹이 뒤에서 달리고 있다.'
score02 = cal_score(embeddings[0][0], embeddings[2][0])  # 23.21
# '치타가 들판을 가로 질러 먹이를 쫓는다.' @ '원숭이 한 마리가 드럼을 연주한다.'
```

## Update history
** Updates on Oct.21.2022 **
- Upload KoSimCSE-unsupervised performance

** Updates on Jun.01.2022 **
- Release KoSimCSE-multitask models

** Updates on May.23.2022 **
- Upload KoSentenceT5 training code
- Upload KoSentenceT5 performance

** Updates on Mar.01.2022 **
- Release KoSimCSE

** Updates on Feb.11.2022 **
- Upload KoSimCSE training code
- Upload KoSimCSE performance

** Updates on Jan.26.2022 **
- Upload KoSBERT training code
- Upload KoSBERT performance

## Baseline Models
Baseline models used for korean sentence embedding - [KLUE-PLMs](https://github.com/KLUE-benchmark/KLUE/blob/main/README.md)

| Model                | Embedding size | Hidden size | # Layers | # Heads |
|----------------------|----------------|-------------|----------|---------|
| KLUE-BERT-base            | 768            | 768         | 12       | 12      |
| KLUE-RoBERTa-base         | 768            | 768         | 12       | 12      |

> **Warning** <br>
> Large pre-trained models need a lot of GPU memory to train

## Available Models
1. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks [[SBERT]-[EMNLP 2019]](https://arxiv.org/abs/1908.10084)
2. SimCSE: Simple Contrastive Learning of Sentence Embeddings [[SimCSE]-[EMNLP 2021]](https://arxiv.org/abs/2104.08821)
3. Sentence-T5: Scalable Sentence Encoders from Pre-trained Text-to-Text Models [[Sentence-T5]-[ACL findings 2022]](https://arxiv.org/abs/2108.08877)

## Datasets
- [kakaobrain KorNLU Datasets](https://github.com/kakaobrain/KorNLUDatasets) (Supervised setting)
- [wiki-corpus](https://github.com/jeongukjae/korean-wikipedia-corpus) (Unsupervised setting)

## Setups
[![Python](https://img.shields.io/badge/python-3.8.5-blue?logo=python&logoColor=FED643)](https://www.python.org/downloads/release/python-385/)
[![Pytorch](https://img.shields.io/badge/pytorch-1.7.1-red?logo=pytorch)](https://pytorch.org/get-started/previous-versions/)

### KoSentenceBERT
- 🤗 [Model Training](https://github.com/BM-K/Sentence-Embedding-is-all-you-need/tree/main/KoSBERT)
- Dataset (Supervised)
    - Training: snli_1.0_train.ko.tsv, sts-train.tsv (multi-task)
      - Performance can be further improved by adding multinli data to training.
    - Validation: sts-dev.tsv
    - Test: sts-test.tsv

### KoSimCSE
- 🤗 [Model Training](https://github.com/BM-K/Sentence-Embedding-is-all-you-need/tree/main/KoSimCSE)
- Dataset (Supervised)
    - Training: snli_1.0_train.ko.tsv + multinli.train.ko.tsv (Supervised setting)
    - Validation: sts-dev.tsv
    - Test: sts-test.tsv
- Dataset (Unsupervised)
    - Training: wiki_corpus.txt
    - Validation: sts-dev.tsv
    - Test: sts-test.tsv

### KoSentenceT5
- 🤗 [Model Training](https://github.com/BM-K/Sentence-Embedding-is-all-you-need/tree/main/KoSentenceT5)
- Dataset (Supervised)
    - Training: snli_1.0_train.ko.tsv + multinli.train.ko.tsv
    - Validation: sts-dev.tsv
    - Test: sts-test.tsv

## Performance-supervised

| Model                  | Average | Cosine Pearson | Cosine Spearman | Euclidean Pearson | Euclidean Spearman | Manhattan Pearson | Manhattan Spearman | Dot Pearson | Dot Spearman |
|------------------------|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| KoSBERT<sup>†</sup><sub>SKT</sub>    | 77.40 | 78.81 | 78.47 | 77.68 | 77.78 | 77.71 | 77.83 | 75.75 | 75.22 |
| KoSBERT              | 80.39 | 82.13 | 82.25 | 80.67 | 80.75 | 80.69 | 80.78 | 77.96 | 77.90 |
| KoSRoBERTa           | 81.64 | 81.20 | 82.20 | 81.79 | 82.34 | 81.59 | 82.20 | 80.62 | 81.25 |
| | | | | | | | | |
| KoSentenceBART         | 77.14 | 79.71 | 78.74 | 78.42 | 78.02 | 78.40 | 78.00 | 74.24 | 72.15 |
| KoSentenceT5          | 77.83 | 80.87 | 79.74 | 80.24 | 79.36 | 80.19 | 79.27 | 72.81 | 70.17 |
| | | | | | | | | |
| KoSimCSE-BERT<sup>†</sup><sub>SKT</sub>   | 81.32 | 82.12 | 82.56 | 81.84 | 81.63 | 81.99 | 81.74 | 79.55 | 79.19 |
| KoSimCSE-BERT              | 83.37 | 83.22 | 83.58 | 83.24 | 83.60 | 83.15 | 83.54 | 83.13 | 83.49 |
| KoSimCSE-RoBERTa          | 83.65 | 83.60 | 83.77 | 83.54 | 83.76 | 83.55 | 83.77 | 83.55 | 83.64 |
| | | | | | | | | | |
| KoSimCSE-BERT-multitask              | 85.71 | 85.29 | 86.02 | 85.63 | 86.01 | 85.57 | 85.97 | 85.26 | 85.93 |
| KoSimCSE-RoBERTa-multitask          | 85.77 | 85.08 | 86.12 | 85.84 | 86.12 | 85.83 | 86.12 | 85.03 | 85.99 |

- [KoSBERT<sup>†</sup><sub>SKT</sub>](https://github.com/BM-K/KoSentenceBERT-SKT)
- [KoSimCSE-BERT<sup>†</sup><sub>SKT</sub>](https://github.com/BM-K/KoSimCSE-SKT)

## Performance-unsupervised

| Model                  | Average | Cosine Pearson | Cosine Spearman | Euclidean Pearson | Euclidean Spearman | Manhattan Pearson | Manhattan Spearman | Dot Pearson | Dot Spearman |
|------------------------|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| KoSRoBERTa-base<sup>†</sup>    | N/A | N/A | 48.96 | N/A | N/A | N/A | N/A | N/A | N/A |
| KoSRoBERTa-large<sup>†</sup>    | N/A | N/A | 51.35 | N/A | N/A | N/A | N/A | N/A | N/A |
| | | | | | | | | | |
| KoSimCSE-BERT    | 71.97 | 72.99 | 71.94 | 71.84 | 72.20 | 71.68 | 72.00 | 72.07 | 71.05 |
| KoSimCSE-RoBERTa    | 73.57 | 74.32 | 73.29 | 73.44 | 73.29 | 73.37 | 73.20 | 74.37 | 73.32 |
| | | | | | | | | | |

- [Korean-SRoBERTa<sup>†</sup>](https://arxiv.org/abs/2004.03289)

## Downstream tasks
- KoSBERT: [Semantic Search](https://github.com/BM-K/Sentence-Embedding-is-all-you-need/tree/main/KoSBERT#semantic-search), [Clustering](https://github.com/BM-K/Sentence-Embedding-is-all-you-need/tree/main/KoSBERT#clustering)
- KoSimCSE: [Semantic Search](https://github.com/BM-K/Sentence-Embedding-is-all-you-need/tree/main/KoSimCSE#semantic-search)

## License
This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />

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
