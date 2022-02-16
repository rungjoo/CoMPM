import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
import math
import pandas as pd
import pdb

from transformers import RobertaTokenizer, RobertaModel
from transformers import BertTokenizer, BertModel
from transformers import GPT2Tokenizer, GPT2Model

class ERC_model(nn.Module):
    def __init__(self, model_type, clsNum, last):
        super(ERC_model, self).__init__()
        self.gpu = True
        self.last = last
        
        """Model Setting"""        
        # model_path = '/data/project/rw/rung/model/'+model_type
        model_path = model_type
        if model_type == 'roberta-large':
            self.context_model = RobertaModel.from_pretrained(model_path)
            tokenizer = RobertaTokenizer.from_pretrained(model_path)
        elif model_type == 'bert-large-uncased':
            self.context_model = BertModel.from_pretrained(model_path)
            tokenizer = BertTokenizer.from_pretrained(model_path)
        else:
            self.context_model = GPT2Model.from_pretrained(model_path)
            tokenizer = GPT2Tokenizer.from_pretrained(model_path)
            tokenizer.add_special_tokens({'cls_token': '[CLS]', 'pad_token': '[PAD]'})            
        tokenizer.add_special_tokens(special_tokens)
        self.context_model.resize_token_embeddings(len(tokenizer))
        self.hiddenDim = self.context_model.config.hidden_size        
            
        """score"""
        self.W = nn.Linear(self.hiddenDim, clsNum)

    def forward(self, batch_input_tokens):
        """
            batch_input_tokens: (batch, len)
        """
        if self.last:
            batch_context_output = self.context_model(batch_input_tokens).last_hidden_state[:,-1,:] # (batch, 1024)
        else:
            batch_context_output = self.context_model(batch_input_tokens).last_hidden_state[:,0,:] # (batch, 1024)
        context_logit = self.W(batch_context_output) # (batch, clsNum)        
        return context_logit