# Korean-Sentence-Embedding
🍭 Korean sentence embedding repository. You can download the pre-trained models and inference right away, also it provides environments where individuals can train models.

** Updates on 2022.03.01 **
- Huggingface model porting

** Updates on 2022.02.11 **
- Upload KoSimCSE training code
- Upload KoSimCSE performance

** Updates on 2022.01.26 **
- Upload KoSBERT training code
- Upload KoSBERT performance

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
    - Train: snli_1.0_train.ko.tsv, sts-train.tsv (multi-task)
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
| KoSBERT              | 80.39 | 82.13 | 82.25 | 80.67 | 80.75 | 80.69 | 80.78 | 77.96 | 77.90 |
| KoSRoBERTa           | 81.64 | 81.20 | 82.20 | 81.79 | 82.34 | 81.59 | 82.20 | 80.62 | 81.25 |
| | | | | | | | | |
| KoSimCSE-BERT<sup>†</sup><sub>SKT</sub>   | 81.32 | 82.12 | 82.56 | 81.84 | 81.63 | 81.99 | 81.74 | 79.55 | 79.19 |
| KoSimCSE-BERT              | 83.37 | 83.22 | 83.58 | 83.24 | 83.60 | 83.15 | 83.54 | 83.13 | 83.49 |
| KoSimCSE-RoBERTa          | 83.65 | 83.60 | 83.77 | 83.54 | 83.76 | 83.55 | 83.77 | 83.55 | 83.64 |
| | | | | | | | | | |
| KoSentenceBART         | 77.14 | 79.71 | 78.74 | 78.42 | 78.02 | 78.40 | 78.00 | 74.24 | 72.15 |
| KoSentenceT5          | 77.83 | 80.87 | 79.74 | 80.24 | 79.36 | 80.19 | 79.27 | 72.81 | 70.17 |

- [KoSBERT<sup>†</sup><sub>SKT</sub>](https://github.com/BM-K/KoSentenceBERT-SKT)
- [KoSimCSE-BERT<sup>†</sup><sub>SKT</sub>](https://github.com/BM-K/KoSimCSE-SKT)

## Downstream Tasks
- KoSBERT: [Semantic Search](https://github.com/BM-K/Sentence-Embedding-is-all-you-need/tree/main/KoSBERT#semantic-search), [Clustering](https://github.com/BM-K/Sentence-Embedding-is-all-you-need/tree/main/KoSBERT#clustering)
```
python SemanticSearch.py
python Clustering.py
```
- KoSimCSE: [Semantic Search](https://github.com/BM-K/Sentence-Embedding-is-all-you-need/tree/main/KoSimCSE#semantic-search)
```
python SemanticSearch.py
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
- [ ] New method
