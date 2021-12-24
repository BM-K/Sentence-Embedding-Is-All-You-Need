import numpy
import torch
import logging
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class ModelDataLoader(Dataset):
    def __init__(self, file_path, args, metric, tokenizer, type_):
        self.type = type_
        self.args = args
        self.metric = metric

        """NLI"""
        self.anchor = []
        self.positive = []
        self.negative = []

        """STS"""
        self.label = []
        self.sentence_1 = []
        self.sentence_2 = []

        #  -------------------------------------
        self.bert_tokenizer = tokenizer
        self.file_path = file_path

        """
        [CLS]: 2
        [PAD]: 0
        [UNK]: 1
        """
        self.init_token = self.bert_tokenizer.cls_token
        self.pad_token = self.bert_tokenizer.pad_token
        self.unk_token = self.bert_tokenizer.unk_token

        self.init_token_idx = self.bert_tokenizer.convert_tokens_to_ids(self.init_token)
        self.pad_token_idx = self.bert_tokenizer.convert_tokens_to_ids(self.pad_token)
        self.unk_token_idx = self.bert_tokenizer.convert_tokens_to_ids(self.unk_token)
        
    def load_data(self, type):

        with open(self.file_path) as file:
            lines = file.readlines()

            for line in lines:
                self.data2tensor(line, type)

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
                                         pad_to_max_length="right")
            
            positive = self.bert_tokenizer(positive_sen, 
                                           truncation=True,
                                           return_tensors="pt",
                                           max_length=self.args.max_len,
                                           pad_to_max_length="right")

            negative = self.bert_tokenizer(negative_sen, 
                                           truncation=True,
                                           return_tensors="pt",
                                           max_length=self.args.max_len,
                                           pad_to_max_length="right")
            
            self.anchor.append(anchor)
            self.positive.append(positive)
            self.negative.append(negative)

        else:
            sentence_1, sentence_2, label = split_data

            sentence_1 = self.bert_tokenizer(sentence_1, 
                                             truncation=True,
                                             return_tensors="pt",
                                             max_length=self.args.max_len,
                                             pad_to_max_length="right")

            sentence_2 = self.bert_tokenizer(sentence_2,
                                             truncation=True,
                                             return_tensors="pt",
                                             max_length=self.args.max_len,
                                             pad_to_max_length="right")


            self.sentence_1.append(sentence_1)
            self.sentence_2.append(sentence_2)
            self.label.append(float(label.strip())/5.0)

    def __getitem__(self, index):

        if self.type == 'train':
            inputs = {'anchor': {
                'source': torch.LongTensor(self.anchor[index]['input_ids']),
                'attention_mask': self.anchor[index]['attention_mask'],
                'token_type_ids': torch.LongTensor(self.anchor[index]['token_type_ids'])
                },
                      'positive': {
                'source': torch.LongTensor(self.positive[index]['input_ids']),
                'attention_mask': self.positive[index]['attention_mask'],
                'token_type_ids': torch.LongTensor(self.positive[index]['token_type_ids'])
                },
                      'negative': {
                'source': torch.LongTensor(self.negative[index]['input_ids']),
                'attention_mask': self.negative[index]['attention_mask'],
                'token_type_ids': torch.LongTensor(self.negative[index]['token_type_ids'])
                }}
        else:

            inputs = {'sentence_1': {
                'source': torch.LongTensor(self.sentence_1[index]['input_ids']),
                'attention_mask': self.sentence_1[index]['attention_mask'],
                'token_type_ids': torch.LongTensor(self.sentence_1[index]['token_type_ids'])
                },
                      'sentence_2': {
                'source': torch.LongTensor(self.sentence_2[index]['input_ids']),
                'attention_mask': self.sentence_2[index]['attention_mask'],
                'token_type_ids': torch.LongTensor(self.sentence_2[index]['token_type_ids'])
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


def convert_to_tensor(corpus, tokenizer, device):
    inputs = tokenizer(corpus,
                       truncation=True,
                       return_tensors="pt",
                       max_length=50,
                       pad_to_max_length="right")
    
    embedding = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    token_type_ids = inputs['token_type_ids']
        
    inputs = {'source': torch.LongTensor(embedding).to(device),
              'token_type_ids': torch.LongTensor(token_type_ids).to(device),
              'attention_mask': attention_mask.to(device)}
    
    return inputs


def example_model_setting(model_ckpt, model_name):

    from model.simcse.bert import BERT

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = BERT(AutoModel.from_pretrained(model_name))
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.load_state_dict(torch.load(model_ckpt)['model'])
    model.to(device)
    model.eval()
    
    return model, tokenizer, device


if __name__ == '__main__':
    get_loader('test')
