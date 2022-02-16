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
    def __init__(self, context_type, speaker_type, clsNum, freeze):
        super(ERC_model, self).__init__()
        self.gpu = True
        
        """Model Setting"""        
        # context_model_path = '/data/project/rw/rung/model/'+context_type
        context_model_path = context_type
        if context_type == 'roberta-large':
            self.context_model = RobertaModel.from_pretrained(context_model_path)
            self.context_last = False
            tokenizer = RobertaTokenizer.from_pretrained(model_path)
        elif context_type == 'bert-large-uncased':
            self.context_model = BertModel.from_pretrained(context_model_path)
            self.context_last = False
            tokenizer = BertTokenizer.from_pretrained(model_path)
        else:
            self.context_model = GPT2Model.from_pretrained(context_model_path)
            tokenizer = GPT2Tokenizer.from_pretrained(context_model_path)
            tokenizer.add_special_tokens({'cls_token': '[CLS]', 'pad_token': '[PAD]'})            
            self.context_last = True
        tokenizer.add_special_tokens(special_tokens)
        self.context_model.resize_token_embeddings(len(tokenizer))
        self.context_hiddenDim = self.context_model.config.hidden_size

        # speaker_model_path = '/data/project/rw/rung/model/'+speaker_type
        speaker_model_path = speaker_type
        if speaker_type == 'roberta-large':
            self.speaker_model = RobertaModel.from_pretrained(speaker_model_path)
            self.speaker_last = False
            tokenizer = RobertaTokenizer.from_pretrained(model_path)
        elif speaker_type == 'bert-large-uncased':
            self.speaker_model = BertModel.from_pretrained(speaker_model_path)
            self.speaker_last = False
            tokenizer = BertTokenizer.from_pretrained(model_path)
        else:
            self.speaker_model = GPT2Model.from_pretrained(speaker_model_path)
            tokenizer = GPT2Tokenizer.from_pretrained(speaker_model_path)
            tokenizer.add_special_tokens({'cls_token': '[CLS]', 'pad_token': '[PAD]'})
            self.speaker_last = True
        tokenizer.add_special_tokens(special_tokens)
        self.speaker_model.resize_token_embeddings(len(tokenizer))
        self.speaker_hiddenDim = self.speaker_model.config.hidden_size
        
        zero = torch.empty(2, 1, self.speaker_hiddenDim).cuda()
        self.h0 = torch.zeros_like(zero) # (num_layers * num_directions, batch, hidden_size)
        self.speakerGRU = nn.GRU(self.speaker_hiddenDim, self.speaker_hiddenDim, 2, dropout=0.3) # (input, hidden, num_layer) (BERT_emb, BERT_emb, num_layer)
            
        """score"""
        self.SC = nn.Linear(self.speaker_hiddenDim, self.context_hiddenDim)
        self.W = nn.Linear(self.context_hiddenDim, clsNum)

        """parameters"""
        self.train_params = list(self.context_model.parameters())+list(self.speakerGRU.parameters())+list(self.SC.parameters())+list(self.W.parameters())
        if not freeze:
            self.train_params += list(self.speaker_model.parameters())        

    def forward(self, batch_input_tokens, batch_speaker_tokens):
        """
            batch_input_tokens: (batch, len)
            batch_speaker_tokens: [(speaker_utt_num, len), ..., ]
        """
        if self.context_last: # GPT
            batch_context_output = self.context_model(batch_input_tokens).last_hidden_state[:,-1,:] # (batch, 1024)
        else: # BERT
            batch_context_output = self.context_model(batch_input_tokens).last_hidden_state[:,0,:] # (batch, 1024)
        
        batch_speaker_output = []
        for speaker_tokens in batch_speaker_tokens:
            if speaker_tokens.shape[0] == 0:
                speaker_track_vector = torch.zeros(1, self.speaker_hiddenDim).cuda()
            else:
                if self.speaker_last:
                    speaker_output = self.speaker_model(speaker_tokens.cuda()).last_hidden_state[:,-1,:] # (speaker_utt_num, 1024)
                else:
                    speaker_output = self.speaker_model(speaker_tokens.cuda()).last_hidden_state[:,0,:] # (speaker_utt_num, 1024)
                speaker_output = speaker_output.unsqueeze(1) # (speaker_utt_num, 1, 1024)
                speaker_GRU_output, _ = self.speakerGRU(speaker_output, self.h0) # (speaker_utt_num, 1, 1024) <- (seq_len, batch, output_size)
                speaker_track_vector = speaker_GRU_output[-1,:,:] # (1, 1024)
            batch_speaker_output.append(speaker_track_vector)
        batch_speaker_output = torch.cat(batch_speaker_output, 0) # (batch, 1024)
                   
        final_output = batch_context_output + self.SC(batch_speaker_output)
        context_logit = self.W(final_output) # (batch, clsNum)
        
        return context_logit