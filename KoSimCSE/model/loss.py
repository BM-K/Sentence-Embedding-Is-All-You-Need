import torch
import logging
import numpy as np
import torch.nn as nn
from model.utils import Metric
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances

logger = logging.getLogger(__name__)


class Loss():

    def __init__(self, args):
        self.args = args
        self.cos = nn.CosineSimilarity(dim=-1)
        self.metric = Metric(args)

    def train_loss_fct(self, config, inputs, a, p, n):
         
        positive_similarity = self.cos(a.unsqueeze(1), p.unsqueeze(0)) / self.args.temperature
        negative_similarity = self.cos(a.unsqueeze(1), n.unsqueeze(0)) / self.args.temperature
        cosine_similarity = torch.cat([positive_similarity, negative_similarity], dim=1).to(self.args.device)

        labels = torch.arange(cosine_similarity.size(0)).long().to(self.args.device)

        loss = config['criterion'](cosine_similarity, labels)

        return loss

    def evaluation_during_training(self, embeddings1, embeddings2, labels, indicator):

        embeddings1 = embeddings1.cpu().numpy()
        embeddings2 = embeddings2.cpu().numpy()
        labels = labels['value'].cpu().numpy().flatten()

        cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
        manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
        euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)
        dot_products = [np.dot(emb1, emb2) for emb1, emb2 in zip(embeddings1, embeddings2)]

        eval_pearson_cosine, _ = pearsonr(labels, cosine_scores)
        eval_spearman_cosine, _ = spearmanr(labels, cosine_scores)

        eval_pearson_manhattan, _ = pearsonr(labels, manhattan_distances)
        eval_spearman_manhattan, _ = spearmanr(labels, manhattan_distances)

        eval_pearson_euclidean, _ = pearsonr(labels, euclidean_distances)
        eval_spearman_euclidean, _ = spearmanr(labels, euclidean_distances)

        eval_pearson_dot, _ = pearsonr(labels, dot_products)
        eval_spearman_dot, _ = spearmanr(labels, dot_products)

        score = {'eval_pearson_cosine': eval_pearson_cosine,
                 'eval_spearman_cosine': eval_spearman_cosine,
                 'eval_pearson_manhattan': eval_pearson_manhattan,
                 'eval_spearman_manhattan': eval_spearman_manhattan,
                 'eval_pearson_euclidean': eval_pearson_euclidean,
                 'eval_spearman_euclidean': eval_spearman_euclidean,
                 'eval_pearson_dot': eval_pearson_dot,
                 'eval_spearman_dot': eval_spearman_dot}

        self.metric.update_indicator(indicator, score)

        return max(eval_spearman_cosine, eval_spearman_manhattan, eval_spearman_euclidean, eval_spearman_dot)
