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

from utils import make_batch_roberta, make_batch_bert, make_batch_gpt
from utils import make_batch_roberta_bert, make_batch_roberta_gpt, make_batch_bert_roberta

def CELoss(pred_outs, labels):
    """
        pred_outs: [batch, clsNum]
        labels: [batch]
    """
    loss = nn.CrossEntropyLoss()
    loss_val = loss(pred_outs, labels)
    return loss_val

    
## finetune RoBETa-large
def main():    
    """Dataset Loading"""
    batch_size = args.batch
    dataset = args.dataset
    dataclass = args.cls
    sample = args.sample
    context_type = args.context_type
    speaker_type = args.speaker_type
    freeze = args.freeze
    
    dataType = 'multi'
    if dataset == 'MELD':
        if args.dyadic:
            dataType = 'dyadic'
        else:
            dataType = 'multi'
        data_path = '../dataset/MELD/'+dataType+'/'
        DATA_loader = MELD_loader
    elif dataset == 'EMORY':
        data_path = '../dataset/EMORY/'
        DATA_loader = Emory_loader
    elif dataset == 'iemocap':
        data_path = '../dataset/iemocap/'
        DATA_loader = IEMOCAP_loader
    elif dataset == 'dailydialog':
        data_path = '../dataset/dailydialog/'
        DATA_loader = DD_loader     
        
    if context_type == 'roberta-large' and speaker_type == 'bert-large-uncased':
        make_batch = make_batch_roberta_bert
    elif context_type == 'roberta-large' and 'gpt' in speaker_type:
        make_batch = make_batch_roberta_gpt
    elif context_type == 'bert-large-uncased' and speaker_type == 'roberta-large':
        make_batch = make_batch_bert_roberta
    else:
        print('batch error')
        
    if freeze:
        freeze_type = 'freeze'
    else:
        freeze_type = 'no_freeze'        
        
    train_path = data_path + dataset+'_train.txt'
    dev_path = data_path + dataset+'_dev.txt'
    test_path = data_path + dataset+'_test.txt'
            
    train_dataset = DATA_loader(train_path, dataclass)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=make_batch)
    train_sample_num = int(len(train_dataset)*sample)
    
    dev_dataset = DATA_loader(dev_path, dataclass)
    dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=make_batch)
    
    test_dataset = DATA_loader(test_path, dataclass)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=make_batch)
    
    """logging and path"""
    save_path = os.path.join(dataset+'_models', context_type+'_'+speaker_type, freeze_type, dataclass)
    
    print("###Save Path### ", save_path)
    log_path = os.path.join(save_path, 'train.log')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fileHandler = logging.FileHandler(log_path)
    
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)    
    logger.setLevel(level=logging.DEBUG)      
    
    """Model Loading"""
    print('DataClass: ', dataclass, '!!!') # emotion
    clsNum = len(train_dataset.labelList)
    model = ERC_model(context_type, speaker_type, clsNum, freeze)
    model = model.cuda()    
    model.train() 
    
    """Training Setting"""        
    training_epochs = args.epoch
    save_term = int(training_epochs/5)
    max_grad_norm = args.norm
    lr = args.lr
    num_training_steps = len(train_dataset)*training_epochs
    num_warmup_steps = len(train_dataset)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr) # , eps=1e-06, weight_decay=0.01
    optimizer = torch.optim.AdamW(model.train_params, lr=lr) # , eps=1e-06, weight_decay=0.01    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    
    """Input & Label Setting"""
    best_dev_fscore, best_test_fscore = 0, 0
    best_dev_fscore_macro, best_dev_fscore_micro, best_test_fscore_macro, best_test_fscore_micro = 0, 0, 0, 0    
    best_epoch = 0
    for epoch in tqdm(range(training_epochs)):
        model.train() 
        for i_batch, data in enumerate(train_dataloader):
            if i_batch > train_sample_num:
                print(i_batch, train_sample_num)
                break
            
            """Prediction"""
            batch_input_tokens, batch_labels, batch_speaker_tokens = data
            batch_input_tokens, batch_labels = batch_input_tokens.cuda(), batch_labels.cuda()
            
            pred_logits = model(batch_input_tokens, batch_speaker_tokens)

            """Loss calculation & training"""
            loss_val = CELoss(pred_logits, batch_labels)
            
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
        """Dev & Test evaluation"""
        model.eval()
        if dataset == 'dailydialog': # micro & macro
            dev_acc, dev_pred_list, dev_label_list = _CalACC(model, dev_dataloader)
            dev_pre_macro, dev_rec_macro, dev_fbeta_macro, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list, average='macro')
            dev_pre_micro, dev_rec_micro, dev_fbeta_micro, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list, labels=[0,1,2,3,5,6], average='micro') # neutral x
            
            dev_fscore = dev_fbeta_macro+dev_fbeta_micro

            """Best Score & Model Save"""
            if dev_fscore > best_dev_fscore_macro + best_dev_fscore_micro:
                best_dev_fscore_macro = dev_fbeta_macro                
                best_dev_fscore_micro = dev_fbeta_micro
                
                test_acc, test_pred_list, test_label_list = _CalACC(model, test_dataloader)
                test_pre_macro, test_rec_macro, test_fbeta_macro, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average='macro')
                test_pre_micro, test_rec_micro, test_fbeta_micro, _ = precision_recall_fscore_support(test_label_list, test_pred_list, labels=[0,1,2,3,5,6], average='micro') # neutral x                
                
                best_epoch = epoch
                _SaveModel(model, save_path)                
        else: # weight
            dev_acc, dev_pred_list, dev_label_list = _CalACC(model, dev_dataloader)
            dev_pre, dev_rec, dev_fbeta, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list, average='weighted')

            """Best Score & Model Save"""
            if dev_fbeta > best_dev_fscore:
                best_dev_fscore = dev_fbeta
                
                test_acc, test_pred_list, test_label_list = _CalACC(model, test_dataloader)
                test_pre, test_rec, test_fbeta, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average='weighted')                
                
                best_epoch = epoch
                _SaveModel(model, save_path)
        
        logger.info('Epoch: {}'.format(epoch))
        if dataset == 'dailydialog': # micro & macro
            logger.info('Devleopment ## accuracy: {}, macro-fscore: {}, micro-fscore: {}'.format(dev_acc, dev_fbeta_macro, dev_fbeta_micro))
            logger.info('')          
        else:
            logger.info('Devleopment ## accuracy: {}, precision: {}, recall: {}, fscore: {}'.format(dev_acc, dev_pre, dev_rec, dev_fbeta))
            logger.info('')
        
    if dataset == 'dailydialog': # micro & macro
        logger.info('Final Fscore ## test-accuracy: {}, test-macro: {}, test-micro: {}, test_epoch: {}'.format(test_acc, test_fbeta_macro, test_fbeta_micro, best_epoch)) 
    else:
        logger.info('Final Fscore ## test-accuracy: {}, test-fscore: {}, test_epoch: {}'.format(test_acc, test_fbeta, best_epoch))         
    
def _CalACC(model, dataloader):
    model.eval()
    correct = 0
    label_list = []
    pred_list = []
    
    # label arragne
    with torch.no_grad():
        for i_batch, data in enumerate(dataloader):            
            """Prediction"""
            batch_input_tokens, batch_labels, batch_speaker_tokens = data
            batch_input_tokens, batch_labels = batch_input_tokens.cuda(), batch_labels.cuda()
            
            pred_logits = model(batch_input_tokens, batch_speaker_tokens) # (1, clsNum)
            
            """Calculation"""    
            pred_label = pred_logits.argmax(1).item()
            true_label = batch_labels.item()
            
            pred_list.append(pred_label)
            label_list.append(true_label)
            if pred_label == true_label:
                correct += 1
        acc = correct/len(dataloader)
    return acc, pred_list, label_list
        
def _SaveModel(model, path):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, 'model.bin'))
    

if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    """Parameters"""
    parser  = argparse.ArgumentParser(description = "Emotion Classifier" )
    parser.add_argument( "--batch", type=int, help = "batch_size", default = 1)
    
    parser.add_argument( "--epoch", type=int, help = 'training epohcs', default = 10) # 12 for iemocap
    parser.add_argument( "--norm", type=int, help = "max_grad_norm", default = 10)
    parser.add_argument( "--lr", type=float, help = "learning rate", default = 1e-6) # 1e-5
    parser.add_argument( "--sample", type=float, help = "sampling trainign dataset", default = 1.0) # 

    parser.add_argument( "--dataset", help = 'MELD or EMORY or iemocap or dailydialog', default = 'MELD')
    
    parser.add_argument( "--context_type", help = 'roberta-large or bert-large-uncased or gpt2 or gpt2-large or gpt2-medium', default = 'roberta-large')
    parser.add_argument( "--speaker_type", help = 'roberta-large or bert-large-uncased or gpt2 or gpt2-large or gpt2-medium', default = 'bert-large-uncased')
    
    parser.add_argument('-dya', '--dyadic', action='store_true', help='dyadic conversation')
    parser.add_argument('-fr', '--freeze', action='store_true', help='freezing PM')
    parser.add_argument( "--cls", help = 'emotion or sentiment', default = 'emotion')    
        
    args = parser.parse_args()
    
    logger = logging.getLogger(__name__)
    streamHandler = logging.StreamHandler()
    
    main()
    