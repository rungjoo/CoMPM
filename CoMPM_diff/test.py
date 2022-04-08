# -*- coding: utf-8 -*-
from tqdm import tqdm
import os
import random
import torch
import torch.nn as nn

from transformers import RobertaTokenizer
from ERC_dataset import MELD_loader, Emory_loader, IEMOCAP_loader, DD_loader
from model import ERC_model
# from ERCcombined import ERC_model

from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
import pdb
import argparse, logging
from sklearn.metrics import precision_recall_fscore_support

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

# tokenizer = RobertaTokenizer.from_pretrained('/data/project/rw/rung/model/roberta-large/')
tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
def make_batch_window(sessions):
    batch_input, batch_labels = [], []
    for session in sessions:
        data = session[0]
        emo_list = session[1]
        
        context_speaker, context, emotion, sentiment = data
        inputString = ""
        for speaker, utt in zip(context_speaker[-5:], context):
            inputString += '<s' + str(speaker+1) + '> ' # s1, s2, s3...
            inputString += utt + " "
        
        for emo_cand in emo_list:
            concat_string = inputString.strip() + " " + tokenizer.sep_token + " " + emo_cand
            batch_input.append(concat_string)
            if emo_cand == emotion:
                batch_labels.append(1)
            else:
                batch_labels.append(0)
    
    batch_input_tokens = tokenizer(batch_input, padding='longest', return_tensors='pt').input_ids # (batch, text_len, 1024)
    batch_labels = torch.tensor(batch_labels)
    
    return batch_input_tokens, batch_labels

def make_batch(sessions):
    batch_input, batch_labels = [], []
    for session in sessions:
        data = session[0]
        emo_list = session[1]
        
        context_speaker, context, emotion, sentiment = data
        inputString = ""
        for speaker, utt in zip(context_speaker[-5:], context):
            inputString += '<s' + str(speaker+1) + '> ' # s1, s2, s3...
            inputString += utt + " "
        
        for emo_cand in emo_list:
            concat_string = inputString.strip() + " " + tokenizer.sep_token + " " + emo_cand
            batch_input.append(encode_right_truncated(concat_string, tokenizer))
            
            if emo_cand == emotion:
                batch_labels.append(1)
            else:
                batch_labels.append(0)
    
    batch_input_tokens = padding(batch_input, tokenizer)
    batch_labels = torch.tensor(batch_labels)
    
    return batch_input_tokens, batch_labels
    
## finetune RoBETa-large
def main():    
    """Dataset Loading"""
    dataset_list = ['MELD', 'EMORY', 'iemocap', 'dailydialog']
    DATA_loader_list = [MELD_loader, Emory_loader, IEMOCAP_loader, DD_loader]
    dataclass = args.cls
    dataType = 'multi'
    
    """logging and path"""
    save_path = os.path.join("MELD_models", dataclass)

    print("###Save Path### ", save_path)
    log_path = os.path.join(save_path, 'test.log')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fileHandler = logging.FileHandler(log_path)

    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)    
    logger.setLevel(level=logging.DEBUG)    
    
    """Model Loading"""
    print('DataClass: ', dataclass, '!!!') # emotion    
    model = ERC_model()
    modelfile = os.path.join(save_path, 'model.bin')
    model.load_state_dict(torch.load(modelfile))
    model = model.cuda()    
    model.eval()    
    
    for dataset, DATA_loader in zip(dataset_list, DATA_loader_list):
        if dataset == 'MELD':
            data_path = os.path.join('dataset', dataset, dataType)
        else:
            data_path = os.path.join('dataset', dataset)

        dev_path = os.path.join(data_path, dataset+'_dev.txt')
        test_path = os.path.join(data_path, dataset+'_test.txt')

        dev_dataset = DATA_loader(dev_path)
        dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=make_batch)

        test_dataset = DATA_loader(test_path)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=make_batch)

        """Dev & Test evaluation"""
        logger.info('####### ' + dataset + ' #######')
        if dataset == 'dailydialog': # micro & macro
            dev_acc, dev_pred_list, dev_label_list = _CalACC(model, dev_dataloader)
            dev_pre_macro, dev_rec_macro, dev_fbeta_macro, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list, average='macro')
            dev_pre_micro, dev_rec_micro, dev_fbeta_micro, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list, labels=[0,1,2,3,5,6], average='micro') # neutral x

            test_acc, test_pred_list, test_label_list = _CalACC(model, test_dataloader)
            test_pre_macro, test_rec_macro, test_fbeta_macro, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average='macro')
            test_pre_micro, test_rec_micro, test_fbeta_micro, _ = precision_recall_fscore_support(test_label_list, test_pred_list, labels=[0,1,2,3,5,6], average='micro') # neutral x

        else: # weight
            dev_acc, dev_pred_list, dev_label_list = _CalACC(model, dev_dataloader)
            dev_pre, dev_rec, dev_fbeta, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list, average='weighted')

            test_acc, test_pred_list, test_label_list = _CalACC(model, test_dataloader)
            test_pre, test_rec, test_fbeta, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average='weighted')

        if dataset == 'dailydialog': # micro & macro
            logger.info('Fscore ## accuracy: {}, dev-macro: {}, dev-micro: {}, test-macro: {}, test-micro: {}'\
                        .format(dev_acc*100, dev_fbeta_macro, dev_fbeta_micro, test_fbeta_macro, test_fbeta_micro))
        else:
            logger.info('Fscore ## accuracy: {}, dev: {}, test: {}'.format(test_acc*100, dev_fbeta, test_fbeta))
        logger.info('')
    
def _CalACC(model, dataloader):
    eval_model = model.eval()
    correct = 0
    label_list = []
    pred_list = []
    
    # label arragne
    with torch.no_grad():
        for i_batch, data in enumerate(tqdm(dataloader)):
            """Prediction"""
            batch_input_tokens, batch_labels = data
            batch_input_tokens, batch_labels = batch_input_tokens.cuda(), batch_labels.cuda()
            
            pred_logits = model(batch_input_tokens)
            
            """Calculation"""
            pred_label = pred_logits[:,1].argmax(0).item() ## max in true probs
            true_label = batch_labels.argmax(0).item()
            
            pred_list.append(pred_label)
            label_list.append(true_label)
            if pred_label == true_label:
                correct += 1
        acc = correct/len(dataloader)
    return acc, pred_list, label_list   

if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    """Parameters"""
    parser  = argparse.ArgumentParser(description = "Emotion Classifier" )
    parser.add_argument( "--dataset", help = 'MELD or EMORY or iemocap or dailydialog', default = 'MELD')
    
    parser.add_argument( "--pretrained", help = 'roberta-large', default = 'roberta-large')
    parser.add_argument('-dya', '--dyadic', action='store_true', help='dyadic conversation')
    parser.add_argument( "--cls", help = 'emotion or sentiment', default = 'emotion')
        
    args = parser.parse_args()
    
    logger = logging.getLogger(__name__)
    streamHandler = logging.StreamHandler()
    
    main()
    