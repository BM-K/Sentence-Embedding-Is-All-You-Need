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
|-|-|-|-|-|-|-|-|-|
| | | | | | | | | |
| KoSBERT<sup>†</sup><sub>SKT</sub>    | 78.81 | 78.47 | 77.68 | 77.78 | 77.71 | 77.83 | 75.75 | 75.22 |
| KoSBERT<sub>base</sub>               |-|-|-|-|-|-|-|-|
| KoSRoBERTa<sub>base</sub>            |-|-|-|-|-|-|-|-|
| | | | | | | | | |
| KoSimCSE-BERT<sup>†</sup><sub>SKT</sub>   | 82.12 | 82.56 | 81.84 | 81.63 | 81.99 | 81.74 | 79.55 | 79.19 |
| KoSimCSE-BERT<sub>base</sub>              |-|-|-|-|-|-|-|-|
| KoSimCSE-RoBERTa<sub>base</sub>           |-|-|-|-|-|-|-|-|

- [KoSBERT<sup>†</sup><sub>SKT</sub>](https://github.com/BM-K/KoSentenceBERT-SKT)
- [KoSimCSE-BERT<sup>†</sup><sub>SKT</sub>](https://github.com/BM-K/KoSimCSE-SKT)
