import sys 
import argparse
import spacy
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
from debias_utils import mask_tokens_for_RE, RE_input_pipeline, format_time, save_model_for_trade_off, mask_tokens
from CDA.substitutor import Substitutor, load_json_pairs

import json

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def save_model(tokenizer, model, file_name):
    tokenizer.save_pretrained(file_name)
    model.save_pretrained(file_name)
    print("saving model to {}".format(file_name))

    return  

def get_loss(outputs, e1, e2, lambda_for_ER, printed):

    e1 = e1.hidden_states[-1].mean(dim = 1)
    e2 = e2.hidden_states[-1].mean(dim = 1)
    # assert list(e1.shape) == list(e2.shape) == [1, 1024]

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    # cos_dis = cos(e1, e2)
    cos_dis = cos(e1, e2)
    # assert list(cos_dis.shape) == [1]

    loss = lambda_for_ER * (-cos_dis.mean()) + outputs.loss

    # if not printed:
    #     logging.info("Current loss is -cos_dis.mean() = {}, outputs.loss = {}".format(-cos_dis.mean(), outputs.loss))
    #     printed = True
        
    return loss

def fine_tune(model, dataloader, eval_type, tokenizer, lr, lambda_for_ER, device, ER_res, epochs = 3):
    model.to(device)
    model.train()

    logging.info("* The learning rate is : {}".format(lr))
    logging.info("* The lambda is : {}".format(lambda_for_ER))

    optimizer = AdamW(model.parameters(),
                      lr=lr,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8)  # args.adam_epsilon  - default is 1e-8.

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)

    logging.info("* The total epoch is : {}".format(str(epochs)))

    # Store the average loss after each epoch so we can plot them.
    loss_values = []

    printed = False

    for epoch_i in range(0, epochs):
        logging.info("* The current epoch is : {}".format(str(epoch_i)))

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0

        for step, batch in enumerate(dataloader):

            b_ids_for_embedding = batch[0].clone()
            b_ids_for_embedding = b_ids_for_embedding.to(device)

            b_input_ids, b_labels = mask_tokens(batch[0], tokenizer)

            b_input_ids = b_input_ids.to(device)
            b_labels = b_labels.to(device)
            b_input_mask = batch[1].to(device)

            b_flipped_input_ids = batch[2].to(device)
            b_flipped_input_mask = batch[3].to(device)

            # indexs = indexs.to(device)

            model.zero_grad()

            outputs = model(b_input_ids,
                            attention_mask=b_input_mask,
                            labels=b_labels, 
                            output_hidden_states=True)

            e1 = model(b_ids_for_embedding,
                            attention_mask=b_input_mask,
                            output_hidden_states=True)

            e2 = model(b_flipped_input_ids,
                            attention_mask=b_flipped_input_mask,
                            output_hidden_states=True)

            # print("* loss function is ", get_bleached_loss)
            loss = get_loss(outputs, e1, e2, lambda_for_ER, printed)
            loss = loss.mean()

            total_loss += loss.item()
            # print("total_loss is : {}, loss.item() type is {}, loss.item() is {}".format(total_loss, type(loss.item()), loss.item()))
            # Perform a backward pass to calculate the gradients.
            loss.backward()
            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the 'exploding gradients' problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            # print([x.grad for x in optimizer.param_groups[0]['params']])
            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)

        # Store the loss value for plotting the learning curve.
        ER_res['loss'].append(avg_train_loss)
        logging.info("* For epoch {} ER_res is : {}".format(epochs, ER_res))

    return model, loss_values

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', required=True)
    parser.add_argument('--lambda_for_ER', required=True, default = 1.0)
    parser.add_argument('--eval_type', required=True, default = 50)
    parser.add_argument('--lr', required=True, default = 5e-4)
    parser.add_argument('--batch_size', required=True, default = 1)
    parser.add_argument('--model', help='which BERT model to use', required=False, default = 'bert-base-uncased')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    max_len_tune = 128

    fine_tune_text_file = "../data/13436_flipped_gap_sents.pkl"
    with open(fine_tune_text_file, 'rb') as f:
        tune_data = pickle.load(f)

    logging.info("* {} sentences are used for finetuning".format(len(tune_data)))

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    args = parse_arguments()
    batch_size = int(args.batch_size)
    logging.info("* batch size is : {}".format(batch_size))
    lr = float(args.lr)
    eval_type = args.eval_type
    epochs = int(args.epochs)
    lambda_for_ER = float(args.lambda_for_ER)

    spacy.load('en_core_web_lg')
    base_pairs = load_json_pairs('./CDA/gender_name_pairs/cda_default_pairs.json')
    name_pairs = load_json_pairs('./CDA/gender_name_pairs/names_pairs_1000_scaled.json')
    substitutor = Substitutor(base_pairs, name_pairs=name_pairs)

    ER_res = {}
    ER_res['loss'] = []

    ER_res['err_nums'] = []
    ER_res['IAs'] = []

    ER_res['seat_ezs'] = []
    ER_res['log_ezs'] = []
    ER_res['ori_seat_ezs'] = []

    model_size = 'base'
    if 'base' in args.model:
        model_size = 'large'

    tokenizer = BertTokenizer.from_pretrained(args.model)
    
    input_ids, attention_masks, flipped_input_ids, flipped_attention_masks \
     = RE_input_pipeline(tune_data, tokenizer, max_len_tune, substitutor)

    train_data = TensorDataset(input_ids, attention_masks, flipped_input_ids, flipped_attention_masks)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    pretrained_model = args.model
    logging.info("* used original model is {}".format(pretrained_model))

    model = BertForMaskedLM.from_pretrained(pretrained_model,
                                            output_attentions=False,
                                            output_hidden_states=False)

    # print('-- Set up model fine-tuning --')
    model, loss_values = fine_tune(model, train_dataloader, eval_type, tokenizer, lr, lambda_for_ER, device, ER_res, epochs = epochs)

    save_model(tokenizer, model, 'saved_models/ER/ER_{}_bert_{}_{}_{}_{}'.format(model_size, eval_type, batch_size, lr, lambda_for_ER))

    # with open('large_res_for_opt_paras/ER/ER_{}_bert_{}_{}_{}_{}.json'.format(model_size, eval_type, batch_size, lr, lambda_for_ER), 'w') as f:
    #     json.dump(ER_res, f, indent = '\t')