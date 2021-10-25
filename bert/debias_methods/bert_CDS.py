import sys 
from CDA.substitutor import Substitutor, load_json_pairs
import argparse
import math
import random
import time
import numpy as np
import pandas as pd
import torch
from nltk import sent_tokenize
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from transformers import BertTokenizer, BertForMaskedLM, AdamW, get_linear_schedule_with_warmup
import pickle
from debias_utils import mask_tokens, CDS_input_pipeline, format_time, save_model_for_trade_off

import spacy

import json

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def fine_tune(model, dataloader, tokenizer, device, CDS_res, lr, eval_type, epochs = 3):
    model.to(device)
    model.train()

    logging.info("* learning rate is {}".format(lr))
    optimizer = AdamW(model.parameters(),
                      lr=lr,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8)  # args.adam_epsilon  - default is 1e-8.

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(dataloader) * epochs
    logging.info("* the total number of batch is : {}".format(str(len(dataloader))))
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)

    logging.info("* The total epoch is : {}".format(str(epochs)))
    # Store the average loss after each epoch so we can plot them.
    loss_values = []

    for epoch_i in range(0, epochs):

        logging.info("* The current epoch is : {}".format(str(epoch_i)))
        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0

        for step, batch in enumerate(dataloader):

            # mask inputs so the model can actually learn something
            b_input_ids, b_labels = mask_tokens(batch[0], tokenizer)
            b_input_ids = b_input_ids.to(device)
            b_labels = b_labels.to(device)
            b_input_mask = batch[1].to(device)

            model.zero_grad()

            outputs = model(b_input_ids,
                            attention_mask=b_input_mask,
                            labels=b_labels)

            # The call to `model` always returns a tuple, so we need to pull the
            # loss value out of the tuple.
            loss = outputs[0]

            total_loss += loss.item()
            # Perform a backward pass to calculate the gradients.
            loss.backward()
            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the 'exploding gradients' problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(dataloader)

        CDS_res['loss'].append(avg_train_loss)
        logging.info("* For epoch {} CDS_res is : {}".format(epochs, CDS_res))
        
    logging.info('Fine-tuning complete!')

    return model, loss_values

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--CDS_ratio', required=True)
    parser.add_argument('--eval_type', required=True)
    parser.add_argument('--lr', required=True, default = 5e-4)
    parser.add_argument('--batch_size', required=True, default = 1)
    parser.add_argument('--epochs', required=True)
    parser.add_argument('--model', help='which BERT model to use', required=False, default = 'bert-base-uncased')
    # parser.add_argument('--output_model', help='output_model', required=False, default = 'test_trade_off/CDS_13436_flipped_gap')
    args = parser.parse_args()
    return args

def save_model(tokenizer, model, file_name):
    tokenizer.save_pretrained(file_name)
    model.save_pretrained(file_name)
    print("saving model to {}".format(file_name))

    return  

if __name__ == '__main__':

    max_len_tune = 128

    args = parse_arguments()
    eval_type = args.eval_type
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    lr = float(args.lr)
    logging.info("* batch_size is {}".format(batch_size))

    ############

    spacy.load('en_core_web_lg')
    base_pairs = load_json_pairs('./CDA/gender_name_pairs/cda_default_pairs.json')
    name_pairs = load_json_pairs('./CDA/gender_name_pairs/names_pairs_1000_scaled.json')
    substitutor = Substitutor(base_pairs, name_pairs=name_pairs)

    fine_tune_text_file = "../data/13436_ori_gap_sents.pkl"
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

    CDS_res = {}
    CDS_res['loss'] = []

    CDS_res['err_nums'] = []
    CDS_res['IAs'] = []

    CDS_res['seat_ezs'] = []
    CDS_res['log_ezs'] = []
    CDS_res['ori_seat_ezs'] = []

    tokenizer = BertTokenizer.from_pretrained(args.model)
    tune_tokens, tune_attentions = CDS_input_pipeline(tune_data, tokenizer, max_len_tune)

    assert tune_tokens.shape == tune_attentions.shape
    train_data = TensorDataset(tune_tokens, tune_attentions)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    pretrained_model = args.model
    logging.info("* used original model is {}".format(pretrained_model))

    model_size = 'base'
    if 'base' in pretrained_model:
        model_size = 'large'

    model = BertForMaskedLM.from_pretrained(pretrained_model,
                                            output_attentions=False,
                                            output_hidden_states=False)

    st = time.time()

    print('-- Set up model fine-tuning --')
    model, loss_values = fine_tune(model, train_dataloader, tokenizer, device, CDS_res, lr, eval_type, epochs = epochs)

    et = time.time()
    logging.info('CDS took {0:.2f} minutes'.format((et - st) / 60))

    save_model(tokenizer, model, 'saved_models/CDS/CDS_{}_bert_{}_{}_{}_{}_{}'.format(model_size, args.CDS_ratio, eval_type, epochs, batch_size, args.lr))

    # with open('saved_models/CDS/CDS_{}_bert_trade_off_{}_{}_{}_{}_{}.json'.format(model_size, args.CDS_ratio, eval_type, epochs, batch_size, args.lr), 'w') as f:
    #     json.dump(CDS_res, f, indent = '\t')

        