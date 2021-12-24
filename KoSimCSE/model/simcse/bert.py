import torch
from torch import nn


class BERT(nn.Module):
    def __init__(self, bert):
        super(BERT, self).__init__()
        self.bert = bert

    def forward(self, config, inputs, mode):

        if mode == 'train':
            
            _, anchor_pooler = self.bert(input_ids=inputs['anchor']['source'],
                                         token_type_ids=inputs['anchor']['token_type_ids'],
                                         attention_mask=inputs['anchor']['attention_mask'],
                                         return_dict=False)
        
            _, positive_pooler = self.bert(input_ids=inputs['positive']['source'],
                                           token_type_ids=inputs['positive']['token_type_ids'],
                                           attention_mask=inputs['positive']['attention_mask'],
                                           return_dict=False)

            _, negative_pooler = self.bert(input_ids=inputs['negative']['source'],
                                           token_type_ids=inputs['negative']['token_type_ids'],
                                           attention_mask=inputs['negative']['attention_mask'],
                                           return_dict=False)

            return anchor_pooler, positive_pooler, negative_pooler

        else:
            _, sentence_1_pooler = self.bert(input_ids=inputs['sentence_1']['source'],
                                             token_type_ids=inputs['sentence_1']['token_type_ids'],
                                             attention_mask=inputs['sentence_1']['attention_mask'],
                                             return_dict=False)

            _, sentence_2_pooler = self.bert(input_ids=inputs['sentence_2']['source'],
                                             token_type_ids=inputs['sentence_2']['token_type_ids'],
                                             attention_mask=inputs['sentence_2']['attention_mask'],
                                             return_dict=False)

            return sentence_1_pooler, sentence_2_pooler

    def encode(self, inputs, device):
    
        _, embeddings = self.bert(input_ids=inputs['source'].to(device),
                                  token_type_ids=inputs['token_type_ids'].to(device),
                                  attention_mask=inputs['attention_mask'].to(device),
                                  return_dict=False)

        return embeddings
