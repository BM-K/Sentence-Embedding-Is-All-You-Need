import os
import logging
from apex import amp
import torch.nn as nn
from tqdm import tqdm
import torch.quantization
import torch.optim as optim
from model.loss import Loss
from model.utils import Metric
from transformers import AutoModel
from model.simcse.bert import BERT
from data.dataloader import get_loader
from transformers import get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


class Processor():

    def __init__(self, args):
        self.args = args
        self.config = None
        self.metric = Metric(args)
        self.loss = Loss(args)
        self.total_steps = 0
        self.model_checker = {'early_stop': False,
                              'early_stop_patient': 0,
                              'best_valid_score': 0}
        self.dev_progress = {'score': 0, 'iter': 0}
        self.model_progress = {'loss': 0, 'iter': 0}

    def run(self, inputs, indicator=None, type=None):

        if type == 'train':
            anchor_embeddings, positive_embeddings, negative_embeddings = self.config['model'](self.config, inputs, type)
            loss = self.loss.train_loss_fct(self.config,
                                            inputs, 
                                            anchor_embeddings, 
                                            positive_embeddings, 
                                            negative_embeddings)
            return loss
        else:
            sentence_1_embeddings, sentence_2_embeddings = self.config['model'](self.config, inputs, type)
            score = self.loss.evaluation_during_training(sentence_1_embeddings,
                                                         sentence_2_embeddings,
                                                         inputs['label'],
                                                         indicator)
            return score

    def progress(self, loss):
        self.model_progress['loss'] += loss
        self.model_progress['iter'] += 1

    def progress_validation(self, score):
        self.dev_progress['score'] += score
        self.dev_progress['iter'] += 1

    def return_value(self):
        loss = self.model_progress['loss'].data.cpu().numpy() / self.model_progress['iter']
        acc = self.model_progress['acc'].data.cpu().numpy() / self.model_progress['iter']

        return loss, acc

    def get_object(self, tokenizer, model):

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.args.lr)

        return criterion, optimizer

    def get_scheduler(self, optim, train_loader):
        train_total = len(train_loader) * self.args.epochs
        scheduler = get_linear_schedule_with_warmup(optim,
                                                    num_warmup_steps=self.args.warmup_ratio * train_total,
                                                    num_training_steps=train_total)

        return scheduler, train_total

    def model_setting(self):
        loader, tokenizer = get_loader(self.args, self.metric)
        model = BERT(AutoModel.from_pretrained(self.args.model))
        model.to(self.args.device)

        criterion, optimizer = self.get_object(tokenizer, model)

        if self.args.train == 'True':
            scheduler, total_steps = self.get_scheduler(optimizer, loader['train'])
            self.total_steps = total_steps
        else:
            scheduler = None

        config = {'loader': loader,
                  'optimizer': optimizer,
                  'criterion': criterion,
                  'scheduler': scheduler,
                  'tokenizer': tokenizer,
                  'args': self.args,
                  'model': model}

        if config['args'].fp16 == 'True':
            config['model'], config['optimizer'] = amp.initialize(
                config['model'], config['optimizer'], opt_level=config['args'].opt_level)

        self.config = config

        return self.config

    def train(self, epoch):
        self.config['model'].train()

        for step, batch in enumerate(tqdm(self.config['loader']['train'])):
            self.config['optimizer'].zero_grad()

            inputs = batch

            train_loss = self.run(inputs, type='train')

            if self.args.fp16 == 'True':
                with amp.scale_loss(train_loss, self.config['optimizer']) as scaled_loss:
                    scaled_loss.backward()
            else:
                train_loss.backward()

            self.config['optimizer'].step()
            self.config['scheduler'].step()

            self.progress(train_loss.data)

            if self.model_progress['iter'] % self.args.eval_steps == 0 or self.model_progress['iter'] == self.total_steps:
                valid_score = self.valid()
                performance = {'tl': train_loss, 'vs': valid_score, 'ep': epoch, 'step': self.model_progress['iter']}
                
                self.metric.save_model(self.config, performance, self.model_checker)
                self.config['model'].train()
                
    def valid(self):
        self.config['model'].eval()
        self.dev_progress = self.dev_progress.fromkeys(self.dev_progress, 0)

        score_indicator = {'eval_pearson_cosine': 0,
                           'eval_spearman_cosine': 0,
                           'eval_pearson_manhattan': 0,
                           'eval_spearman_manhattan': 0,
                           'eval_pearson_euclidean': 0,
                           'eval_spearman_euclidean': 0,
                           'eval_pearson_dot': 0,
                           'eval_spearman_dot': 0}

        with torch.no_grad():
            for step, batch in enumerate(self.config['loader']['valid']):
                inputs = batch
                score = self.run(inputs, indicator=score_indicator, type='valid')

                self.progress_validation(score)

        score = self.metric.cal_dev_score(self.dev_progress, score_indicator)

        return score

    def test(self):
        self.config['model'].load_state_dict(torch.load(self.args.path_to_saved_model)['model'], strict=False)
        self.config['model'].eval()
        self.dev_progress = self.dev_progress.fromkeys(self.dev_progress, 0)

        score_indicator = {'eval_pearson_cosine': 0,
                           'eval_spearman_cosine': 0,
                           'eval_pearson_manhattan': 0,
                           'eval_spearman_manhattan': 0,
                           'eval_pearson_euclidean': 0,
                           'eval_spearman_euclidean': 0,
                           'eval_pearson_dot': 0,
                           'eval_spearman_dot': 0}

        with torch.no_grad():
            for step, batch in enumerate(self.config['loader']['test']):
                inputs = batch
                score = self.run(inputs, indicator=score_indicator, type='test')

                self.progress_validation(score)

        logger.info('### TEST SCORE ###')
        score = self.metric.cal_dev_score(self.dev_progress, score_indicator)
