import sys 
import argparse
import math
import random
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset, RandomSampler, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
import pickle
import datetime

import spacy
from CDA.substitutor import Substitutor, load_json_pairs

import json

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

class MyDataset(Dataset):

    def __init__(self, txt_list, tokenizer, gpt2_type="gpt2", max_length=128):

        self.tokenizer = tokenizer # the gpt2 tokenizer we instantiated
        self.input_ids = []
        self.attn_masks = []

        for txt in txt_list:
            # encodings_dict = tokenizer(txt)
            # encodings_dict = tokenizer('<|startoftext|>'+ txt + '<|endoftext|>', 
            #                                                      pad_to_max_length=True)
            encodings_dict = tokenizer('<|startoftext|>'+ txt + '<|endoftext|>', 
                                                                 truncation=True, 
                                                                 max_length=max_length, 
                                                                 padding="max_length")
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
        
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx] 

def fine_tune(model, dataloader, tokenizer, device, CDS_res, lr, eval_type, epochs = 3):

    model.to(device)
    model.train()

    warmup_steps = 1e2

    logging.info("* learning rate is {}".format(lr))
    optimizer = AdamW(model.parameters(),
                      lr = lr,
                      eps = 1e-8
                    )

    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = warmup_steps, 
                                                num_training_steps = total_steps)

    
    logging.info("* The total epoch is : {}".format(str(epochs)))

    for epoch_i in range(0, epochs):
        logging.info("* The current epoch is : {}".format(str(epoch_i)))
        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0

        for step, batch in enumerate(dataloader):

            # mask inputs so the model can actually learn something
            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_masks = batch[1].to(device)

            model.zero_grad()        

            outputs = model(  b_input_ids,
                              labels=b_labels, 
                              attention_mask = b_masks,
                              token_type_ids=None
                            )

            loss = outputs[0]  

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(dataloader)
        CDS_res['loss'].append(avg_train_loss)

    return model

def save_model(tokenizer, model, file_name):
    tokenizer.save_pretrained(file_name)
    model.save_pretrained(file_name)
    print("saving model to {}".format(file_name))

    return   

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--CDS_ratio', required=True)
    parser.add_argument('--eval_type', required=True)
    parser.add_argument('--lr', required=True, default = 5e-4)
    parser.add_argument('--batch_size', required=True, default = 1)
    parser.add_argument('--epochs', required=True)
    parser.add_argument('--model', help='which BERT model to use', required=False, default = 'bert-base-uncased')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_arguments()
    CDS_ratio = float(args.CDS_ratio)
    eval_type = args.eval_type
    batch_size = int(args.batch_size)
    lr = float(args.lr)
    epochs = int(args.epochs)
    pretrained_model = args.model
    
    logging.info("* batch_size is {}".format(batch_size))

    CDS_res = {}
    CDS_res['loss'] = []
    CDS_res['seat_ezs'] = []
    CDS_res['ori_seat_ezs'] = []
    CDS_res['IAs'] = []
    CDS_res['err_nums'] = []

    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model, 
    #                                               bos_token='<|startoftext|>', 
    #                                               eos_token='<|endoftext|>', 
    #                                               pad_token='<|pad|>')

    ############

    spacy.load('en_core_web_lg')
    base_pairs = load_json_pairs('./CDA/gender_name_pairs/cda_default_pairs.json')
    name_pairs = load_json_pairs('./CDA/gender_name_pairs/names_pairs_1000_scaled.json')
    substitutor = Substitutor(base_pairs, name_pairs=name_pairs)

    fine_tune_text_file = "../data/collected_sents/13436_ori_gap_sents.pkl"
    with open(fine_tune_text_file, 'rb') as f:
        tune_data = pickle.load(f)
    logging.info("* {} sentences are used for finetuning".format(len(tune_data)))

    CDS_ratio = float(args.CDS_ratio)
    size = int(CDS_ratio * len(tune_data))
    logging.info("* CDS_ratio : {}, the number of correponding inverted sentences {}".format(CDS_ratio, size))

    sents_A = tune_data[:size]
    sents_A = [substitutor.invert_document(sent)[0] for sent in sents_A]

    logging.info(sents_A[:3])

    sents_B = tune_data[size:]

    assert len(sents_A + sents_B) == len(tune_data)

    tune_data = sents_A + sents_B

    ############


    dataset = MyDataset(tune_data, tokenizer)

    dataloader = DataLoader(
            dataset,  
            sampler = RandomSampler(dataset), # Sampling for training is random
            batch_size = batch_size
        )

    model_size = ''
    if 'medium' in pretrained_model:
        model_size = 'medium'

    model = GPT2LMHeadModel.from_pretrained(pretrained_model)
    # model.resize_token_embeddings(len(tokenizer))

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    st = time.time()

    print('-- Set up model fine-tuning --')
    model = fine_tune(model, dataloader, tokenizer, device, CDS_res, lr, eval_type, epochs = epochs)

    et = time.time()
    logging.info('CDS took {0:.2f} minutes'.format((et - st) / 60))

    save_model(tokenizer, model, 'saved_models/CDS/CDS_{}_gpt2_{}_{}_{}_{}_{}'.format(model_size, CDS_ratio, eval_type, epochs, batch_size, lr))

    # with open('large_res_for_opt_paras/CDS/CDS_gpt2_trade_off_{}_{}_{}_{}.json'.format(CDS_ratio, eval_type, epochs, batch_size, lr), 'w') as f:
    #     json.dump(CDS_res, f, indent = '\t')