import torch
import random
import logging
import numpy as np
from argparse import ArgumentParser


class Arguments():

    def __init__(self):
        self.parser = ArgumentParser()

    def add_type_of_processing(self):
        self.add_argument('--opt_level', type=str, default='O1')
        self.add_argument('--fp16', type=str, default='True')
        self.add_argument('--train', type=str, default='True')
        self.add_argument('--test', type=str, default='True')
        self.add_argument('--device', type=str, default=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    def add_hyper_parameters(self):
        self.add_argument('--model', type=str, default='klue/bert-base')
        self.add_argument('--patient', type=int, default=10)
        self.add_argument('--dropout', type=int, default=0.1)
        self.add_argument('--max_len', type=int, default=50)
        self.add_argument('--batch_size', type=int, default=256)
        self.add_argument('--epochs', type=int, default=3)
        self.add_argument('--eval_steps', type=int, default=250)
        self.add_argument('--seed', type=int, default=12)
        self.add_argument('--lr', type=float, default=0.00005)
        self.add_argument('--weight_decay', type=float, default=0.1)
        self.add_argument('--warmup_ratio', type=float, default=0.05)
        self.add_argument('--temperature', type=float, default=0.05)

    def add_data_parameters(self):
        self.add_argument('--train_data', type=str, default='train_nli.tsv')
        self.add_argument('--valid_data', type=str, default='valid_sts.tsv')
        self.add_argument('--test_data', type=str, default='test_sts.tsv')
        self.add_argument('--task', type=str, default='NLU')
        self.add_argument('--path_to_data', type=str, default='./data/')
        self.add_argument('--path_to_save', type=str, default='./output/')
        self.add_argument('--path_to_saved_model', type=str, default='./output/')
        self.add_argument('--ckpt', type=str, default='best_checkpoint.pt')

    def print_args(self, args):
        for idx, (key, value) in enumerate(args.__dict__.items()):
            if idx == 0:print("argparse{\n", "\t", key, ":", value)
            elif idx == len(args.__dict__) - 1:print("\t", key, ":", value, "\n}")
            else:print("\t", key, ":", value)

    def add_argument(self, *args, **kw_args):
        return self.parser.add_argument(*args, **kw_args)

    def parse(self):
        args = self.parser.parse_args()
        self.print_args(args)

        return args


class Setting():

    def set_logger(self):

        _logger = logging.getLogger()
        formatter = logging.Formatter(
            '[%(levelname)s] %(asctime)s [ %(message)s ] | file::%(filename)s | line::%(lineno)s')

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        _logger.addHandler(stream_handler)
        _logger.setLevel(logging.DEBUG)

        return _logger

    def set_seed(self, args):

        seed = args.seed

        random.seed(seed)
        np.random.seed(seed)

        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def run(self):

        parser = Arguments()
        parser.add_type_of_processing()
        parser.add_hyper_parameters()
        parser.add_data_parameters()

        args = parser.parse()
        logger = self.set_logger()
        self.set_seed(args)

        return args, logger
