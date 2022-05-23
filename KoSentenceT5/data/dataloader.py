import numpy
import torch
import logging
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


class ModelDataLoader(Dataset):
    def __init__(self, file_path, args, metric, tokenizer, type_):
        self.type = type_
        self.args = args
        self.metric = metric

        """NLI"""
        self.anchor = []
        self.anchor_dec = []

        self.positive = []
        self.positive_dec = []

        self.negative = []
        self.negative_dec = []

        """STS"""
        self.label = []
        
        self.sentence_1 = []
        self.sentence_1_dec = []

        self.sentence_2 = []
        self.sentence_2_dec = []

        #  -------------------------------------
        self.bert_tokenizer = tokenizer
        self.file_path = file_path

        special_tokens = {'bos_token': "[CLS]"}
        self.bert_tokenizer.add_special_tokens(special_tokens)

        self.init_token = self.bert_tokenizer.bos_token
        self.pad_token = self.bert_tokenizer.pad_token
        self.unk_token = self.bert_tokenizer.unk_token
        self.eos_token = self.bert_tokenizer.eos_token

        self.init_token_idx = self.bert_tokenizer.convert_tokens_to_ids(self.init_token)
        self.pad_token_idx = self.bert_tokenizer.convert_tokens_to_ids(self.pad_token)
        self.unk_token_idx = self.bert_tokenizer.convert_tokens_to_ids(self.unk_token)
        self.eos_token_idx = self.bert_tokenizer.convert_tokens_to_ids(self.eos_token)
        
        print(self.init_token, self.init_token_idx)
        print(self.pad_token, self.pad_token_idx)
        print(self.unk_token, self.unk_token_idx)
        print(self.eos_token, self.eos_token_idx)
        
    def load_data(self, type):

        with open(self.file_path) as file:
            lines = file.readlines()

            for line in lines:
                _ = self.data2tensor(line, type)
                
        if type == 'train':
            assert len(self.anchor) == len(self.positive) == len(self.negative)
        else:
            assert len(self.sentence_1) == len(self.sentence_2) == len(self.label)

    def data2tensor(self, line, type):
        split_data = line.split('\t')

        if type == 'train':
            anchor_sen, positive_sen, negative_sen = split_data
            
            anchor = self.bert_tokenizer(anchor_sen, 
                                         truncation=True,
                                         return_tensors="pt",
                                         max_length=self.args.max_len,
                                         padding='max_length')
            
            anh_dec_ids = torch.cat([torch.tensor([self.init_token_idx]).unsqueeze(0), anchor['input_ids'][:, :-1]], dim=-1)    
            anchor['dec_ids'] = anh_dec_ids
        
            positive = self.bert_tokenizer(positive_sen, 
                                           truncation=True,
                                           return_tensors="pt",
                                           max_length=self.args.max_len,
                                           padding='max_length')

            pos_dec_ids = torch.cat([torch.tensor([self.init_token_idx]).unsqueeze(0), positive['input_ids'][:, :-1]], dim=-1)
            positive['dec_ids'] = pos_dec_ids
            
            negative = self.bert_tokenizer(negative_sen, 
                                           truncation=True,
                                           return_tensors="pt",
                                           max_length=self.args.max_len,
                                           padding='max_length')
        
            neg_dec_ids = torch.cat([torch.tensor([self.init_token_idx]).unsqueeze(0), negative['input_ids'][:, :-1]], dim=-1)
            negative['dec_ids'] = neg_dec_ids
            
            self.anchor.append(anchor)
            self.positive.append(positive)
            self.negative.append(negative)
        
        else:
            sentence_1, sentence_2, label = split_data
    
            sentence_1 = self.bert_tokenizer(sentence_1, 
                                             truncation=True,
                                             return_tensors="pt",
                                             max_length=self.args.max_len,
                                             padding='max_length')
            
            s1_dec_ids = torch.cat([torch.tensor([self.init_token_idx]).unsqueeze(0), sentence_1['input_ids'][:, :-1]], dim=-1)    
            
            sentence_1['dec_ids'] = s1_dec_ids
            
            sentence_2 = self.bert_tokenizer(sentence_2,
                                             truncation=True,
                                             return_tensors="pt",
                                             max_length=self.args.max_len,
                                             padding='max_length')
            s2_dec_ids = torch.cat([torch.tensor([self.init_token_idx]).unsqueeze(0), sentence_2['input_ids'][:, :-1]], dim=-1)
            
            sentence_2['dec_ids'] = s2_dec_ids
            
            self.sentence_1.append(sentence_1)
            self.sentence_2.append(sentence_2)
            self.label.append(float(label.strip())/5.0)

    def __getitem__(self, index):

        if self.type == 'train':
            inputs = {'anchor': {
                'source': torch.LongTensor(self.anchor[index]['input_ids']),
                'attention_mask': self.anchor[index]['attention_mask'],
                'dec_ids': torch.LongTensor(self.anchor[index]['dec_ids'])
                                },
                      'positive': {
                'source': torch.LongTensor(self.positive[index]['input_ids']),
                'attention_mask': self.positive[index]['attention_mask'],
                'dec_ids': torch.LongTensor(self.positive[index]['dec_ids'])
                                },
                      'negative': {
                'source': torch.LongTensor(self.negative[index]['input_ids']),
                'attention_mask': self.negative[index]['attention_mask'],
                'dec_ids': torch.LongTensor(self.negative[index]['dec_ids'])
                                }}
        else:

            inputs = {'sentence_1': {
                'source': torch.LongTensor(self.sentence_1[index]['input_ids']),
                'attention_mask': self.sentence_1[index]['attention_mask'],
                'dec_ids': torch.LongTensor(self.sentence_1[index]['dec_ids'])
                                },
                      'sentence_2': {
                'source': torch.LongTensor(self.sentence_2[index]['input_ids']),
                'attention_mask': self.sentence_2[index]['attention_mask'],
                'dec_ids': torch.LongTensor(self.sentence_2[index]['dec_ids'])
                                },
                      'label': {
                          'value': torch.FloatTensor([self.label[index]])}
                }

        for key, value in inputs.items():
            for inner_key, inner_value in value.items():
                inputs[key][inner_key] = inner_value.squeeze(0)
                
        inputs = self.metric.move2device(inputs, self.args.device)
        
        return inputs

    def __len__(self):
        if self.type == 'train':
            return len(self.anchor)
        else:
            return len(self.label)


# Get train, valid, test data loader and BERT tokenizer
def get_loader(args, metric):
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    path_to_train_data = args.path_to_data + '/' + args.train_data
    path_to_valid_data = args.path_to_data + '/' + args.valid_data
    path_to_test_data = args.path_to_data + '/' + args.test_data

    if args.train == 'True' and args.test == 'False':
        train_iter = ModelDataLoader(path_to_train_data, args, metric, tokenizer, type_='train')
        valid_iter = ModelDataLoader(path_to_valid_data, args, metric, tokenizer, type_='valid')

        train_iter.load_data('train')
        valid_iter.load_data('valid')

        loader = {'train': DataLoader(dataset=train_iter,
                                      batch_size=args.batch_size,
                                      shuffle=True),
                  'valid': DataLoader(dataset=valid_iter,
                                      batch_size=args.batch_size,
                                      shuffle=True)}

    elif args.train == 'False' and args.test == 'True':
        test_iter = ModelDataLoader(path_to_test_data, args, metric, tokenizer, type_='test')
        test_iter.load_data('test')

        loader = {'test': DataLoader(dataset=test_iter,
                                     batch_size=args.batch_size,
                                     shuffle=True)}

    else:
        loader = None

    return loader, tokenizer


if __name__ == '__main__':
    get_loader('test')
