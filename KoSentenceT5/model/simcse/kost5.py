import torch
from torch import nn
from transformers import BartForSequenceClassification, AutoModel

class KoSentenceT5(nn.Module):
    def __init__(self, model):
        super(KoSentenceT5, self).__init__()
        self.model = AutoModel.from_pretrained(model)
    
    def forward(self, config, inputs, mode):

        if mode == 'train':
            
            anchor_pooler = self.model(input_ids=inputs['anchor']['source'],
                                       attention_mask=inputs['anchor']['attention_mask'],
                                       decoder_input_ids=inputs['anchor']['dec_ids']
                                       )
        
            positive_pooler = self.model(input_ids=inputs['positive']['source'],
                                         attention_mask=inputs['positive']['attention_mask'],
                                         decoder_input_ids=inputs['positive']['dec_ids']
                                         )
        
            negative_pooler = self.model(input_ids=inputs['negative']['source'],
                                         attention_mask=inputs['negative']['attention_mask'],
                                         decoder_input_ids=inputs['negative']['dec_ids']
                                         )
            
            return anchor_pooler, positive_pooler, negative_pooler

        else:
            sentence_1_pooler = self.model(input_ids=inputs['sentence_1']['source'],
                                           attention_mask=inputs['sentence_1']['attention_mask'],
                                           decoder_input_ids=inputs['sentence_1']['dec_ids']
                                           )
            
            sentence_2_pooler = self.model(input_ids=inputs['sentence_2']['source'],
                                           attention_mask=inputs['sentence_2']['attention_mask'],
                                           decoder_input_ids=inputs['sentence_2']['dec_ids']
                                           )
    
            return sentence_1_pooler, sentence_2_pooler

    def encode(self, inputs, device):
    
        embeddings = self.model(input_ids=inputs['source'].to(device),
                                attention_mask=inputs['attention_mask'].to(device),
                                )

        return ((embeddings * inputs['attention_mask'].unsqueeze(-1)).sum(1) / inputs['attention_mask'].sum(-1).unsqueeze(-1))
