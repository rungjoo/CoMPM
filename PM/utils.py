import torch

from transformers import RobertaTokenizer, RobertaModel
from transformers import BertTokenizer, BertModel
from transformers import GPT2Tokenizer, GPT2Model

# roberta_tokenizer = RobertaTokenizer.from_pretrained('/data/project/rw/rung/model/roberta-large/')
# bert_tokenizer = BertTokenizer.from_pretrained('/data/project/rw/rung/model/bert-large-uncased/')
# gpt_tokenizer = GPT2Tokenizer.from_pretrained('/data/project/rw/rung/model/gpt2-large/')

roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
bert_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
gpt_tokenizer.add_special_tokens({'cls_token': '[CLS]', 'pad_token': '[PAD]'})

def encode_right_truncated(text, tokenizer, max_length=511):
    tokenized = tokenizer.tokenize(text)
    truncated = tokenized[-max_length:]    
    ids = tokenizer.convert_tokens_to_ids(truncated)
    
    return [tokenizer.cls_token_id] + ids

def padding(ids_list, tokenizer):
    max_len = 0
    for ids in ids_list:
        if len(ids) > max_len:
            max_len = len(ids)
    
    pad_ids = []
    for ids in ids_list:
        pad_len = max_len-len(ids)
        add_ids = [tokenizer.pad_token_id for _ in range(pad_len)]
        
        pad_ids.append(ids+add_ids)
    
    return torch.tensor(pad_ids)

def encode_right_truncated_gpt(text, tokenizer, max_length=511):
    tokenized = tokenizer.tokenize(text)
    truncated = tokenized[-max_length:]    
    ids = tokenizer.convert_tokens_to_ids(truncated)
    
    return ids + [tokenizer.cls_token_id]

def padding_gpt(ids_list, tokenizer):
    max_len = 0
    for ids in ids_list:
        if len(ids) > max_len:
            max_len = len(ids)
    
    pad_ids = []
    for ids in ids_list:
        pad_len = max_len-len(ids)
        add_ids = [tokenizer.pad_token_id for _ in range(pad_len)]
        
        pad_ids.append(add_ids+ids)
    
    return torch.tensor(pad_ids)

def make_batch_roberta(sessions):
    batch_input, batch_labels = [], []
    for session in sessions:
        data = session[0]
        label_list = session[1]
        
        utt, emotion, sentiment = data        
        batch_input.append(encode_right_truncated(utt.strip(), roberta_tokenizer))
        
        if len(label_list) > 3:
            label_ind = label_list.index(emotion)
        else:
            label_ind = label_list.index(sentiment)
        batch_labels.append(label_ind)
    
    batch_input_tokens = padding(batch_input, roberta_tokenizer)
    batch_labels = torch.tensor(batch_labels)    
    
    return batch_input_tokens, batch_labels

def make_batch_bert(sessions):
    batch_input, batch_labels = [], []
    for session in sessions:
        data = session[0]
        label_list = session[1]
        
        utt, emotion, sentiment = data        
        batch_input.append(encode_right_truncated(utt.strip(), bert_tokenizer))
        
        if len(label_list) > 3:
            label_ind = label_list.index(emotion)
        else:
            label_ind = label_list.index(sentiment)
        batch_labels.append(label_ind)
    
    batch_input_tokens = padding(batch_input, roberta_tokenizer)
    batch_labels = torch.tensor(batch_labels)    
    
    return batch_input_tokens, batch_labels

def make_batch_gpt(sessions):
    batch_input, batch_labels = [], []
    for session in sessions:
        data = session[0]
        label_list = session[1]
        
        utt, emotion, sentiment = data
        batch_input.append(encode_right_truncated_gpt(utt.strip(), gpt_tokenizer, max_length = 511))
        
        if len(label_list) > 3:
            label_ind = label_list.index(emotion)
        else:
            label_ind = label_list.index(sentiment)
        batch_labels.append(label_ind)
    
    batch_input_tokens = padding(batch_input, roberta_tokenizer)
    batch_labels = torch.tensor(batch_labels)    
    
    return batch_input_tokens, batch_labels