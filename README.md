# Korean-Sentence-Embedding
Korean sentence embedding repository
## Baseline Models
Baseline models used for korean sentence embedding - [KLUE-PLMs](https://github.com/KLUE-benchmark/KLUE/blob/main/README.md)

| Model                | Embedding size | Hidden size | # Layers | # Heads |
|----------------------|----------------|-------------|----------|---------|
| KLUE-BERT-base            | 768            | 768         | 12       | 12      |
| KLUE-RoBERTa-base         | 768            | 768         | 12       | 12      |

`NOTE`:  All the pretrained models are uploaded in Huggingface Model Hub. Check https://huggingface.co/klue.
<br>

## KoSentenceBERT
- [SBERT-Paper](https://arxiv.org/abs/1908.10084)
## KoSimCSE
- [SimCSE-Paper](https://arxiv.org/abs/2104.08821)
## Performance

| Model                  | Cosine Pearson | Cosine Spearman | Euclidean Pearson | Euclidean Spearman | Manhattan Pearson | Manhattan Spearman | Dot Pearson | Dot Spearman |
|------------------------|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| KoSBERT<sup>†</sup><sub>SKT</sub>    | 78.81 | 78.47 | 77.68 | 77.78 | 77.71 | 77.83 | 75.75 | 75.22 |
| KoSBERT<sub>base</sub>               |-|-|-|-|-|-|-|-|
| KoSRoBERTa<sub>base</sub>            |-|-|-|-|-|-|-|-|
| | | | | | | | | |
| KoSimCSE-BERT<sup>†</sup><sub>SKT</sub>   | 82.12 | 82.56 | 81.84 | 81.63 | 81.99 | 81.74 | 79.55 | 79.19 |
| KoSimCSE-BERT<sub>base</sub>              |-|-|-|-|-|-|-|-|
| KoSimCSE-RoBERTa<sub>base</sub>           |-|-|-|-|-|-|-|-|

- [KoSBERT<sup>†</sup><sub>SKT</sub>](https://github.com/BM-K/KoSentenceBERT-SKT)
- [KoSimCSE-BERT<sup>†</sup><sub>SKT</sub>](https://github.com/BM-K/KoSimCSE-SKT)

## References

```
@misc{gao2021simcse,
    title={SimCSE: Simple Contrastive Learning of Sentence Embeddings},
    author={Tianyu Gao and Xingcheng Yao and Danqi Chen},
    year={2021},
    eprint={2104.08821},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

```
@misc{ham2020kornli,
    title={KorNLI and KorSTS: New Benchmark Datasets for Korean Natural Language Understanding},
    author={Jiyeon Ham and Yo Joong Choe and Kyubyong Park and Ilji Choi and Hyungjoon Soh},
    year={2020},
    eprint={2004.03289},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

```
@misc{park2021klue,
    title={KLUE: Korean Language Understanding Evaluation},
    author={Sungjoon Park and Jihyung Moon and Sungdong Kim and Won Ik Cho and Jiyoon Han and Jangwon Park and Chisung Song and Junseong Kim and Yongsook Song and Taehwan Oh and Joohong Lee and Juhyun Oh and Sungwon Lyu and Younghoon Jeong and Inkwon Lee and Sangwoo Seo and Dongjun Lee and Hyunwoo Kim and Myeonghwa Lee and Seongbo Jang and Seungwon Do and Sunkyoung Kim and Kyungtae Lim and Jongwon Lee and Kyumin Park and Jamin Shin and Seonghyun Kim and Lucy Park and Alice Oh and Jung-Woo Ha and Kyunghyun Cho},
    year={2021},
    eprint={2105.09680},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
