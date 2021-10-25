import sys 
import argparse
import spacy
import math
import random
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset, RandomSampler, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
import pickle
from CDA.substitutor import Substitutor, load_json_pairs
import datetime
# from evaluation_for_trade_off.evaluating_over_debiasing_gpt2 import add_predicts, get_vios_and_dif
# from evaluation_for_trade_off.gpt2_ori_seat import get_embeddings, compute_effect_size

# from evaluation_for_trade_off.evaluate_by_divers_tmps_gpt_2 import convert_tmps_to_sents, comput_asso_for_autoregressive, add_assos

# from evaluation_for_trade_off.bias_utils.debiasing_effects_util import get_effect_size
import json
from tqdm import tqdm, trange
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def save_model(tokenizer, model, file_name):
    tokenizer.save_pretrained(file_name)
    model.save_pretrained(file_name)
    print("saving model to {}".format(file_name))

    return   

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

class ERDataset(Dataset):

    def __init__(self, txt_list, tokenizer, substitutor, gpt2_type="gpt2", max_length=128):

        self.tokenizer = tokenizer # the gpt2 tokenizer we instantiated
        self.input_ids1 = []
        self.attn_masks1 = []
        self.input_ids2 = []
        self.attn_masks2 = []

        for txt in txt_list:
            # encodings_dict = tokenizer('<|endoftext|>'+ txt + '<|endoftext|>')
            encodings_dict = tokenizer('<|startoftext|>'+ txt + '<|endoftext|>', 
                                                                 truncation=True, 
                                                                 max_length=max_length, 
                                                                 padding="max_length")
            self.input_ids1.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks1.append(torch.tensor(encodings_dict['attention_mask']))

            flipped_sent, _ = substitutor.invert_document(txt)
            # encodings_dict = tokenizer('<|endoftext|>'+ flipped_sent + '<|endoftext|>')
            encodings_dict = tokenizer('<|startoftext|>'+ flipped_sent + '<|endoftext|>', 
                                                                 truncation=True, 
                                                                 max_length=max_length, 
                                                                 padding="max_length")
            self.input_ids2.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks2.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids1)

    def __getitem__(self, idx):
        return self.input_ids1[idx], self.attn_masks1[idx], self.input_ids2[idx], self.attn_masks2[idx] 

def get_loss(outputs, flipped_outputs, lambda_for_ER):

    # logging.info("*** The shape of outputs.hidden_states[-1][:, -1, :] is {}".format(outputs.hidden_states[-1][:, -1, :].shape))
    # logging.info("*** The shape of flipped_outputs.hidden_states[-1][:, -1, :] is {}".format(flipped_outputs.hidden_states[-1][:, -1, :].shape))

    e1 = outputs.hidden_states[-1][:, -1, :]
    e2 = flipped_outputs.hidden_states[-1][:, -1, :]
    # assert list(e1.shape) == list(e2.shape) == [1, 1024]

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    cos_dis = cos(e1, e2)

    # logging.info("*** The shape of outputs.loss  is {}".format(outputs.loss.shape))
    # logging.info("*** lambda_for_ER  is {}, type is {}".format(lambda_for_ER, type(lambda_for_ER)))

    loss = lambda_for_ER * (-cos_dis.mean()) + outputs.loss

    # logging.info("*** The shape of loss is {}".format(loss.shape))

    return loss

def fine_tune(model, dataloader, eval_type, tokenizer, lr, lambda_for_ER, device, ER_res, epochs = 3):
    model.to(device)
    model.train()

    logging.info("* The total epoch is : {}".format(str(epochs)))
    logging.info("* The learning rate is : {}".format(lr))
    logging.info("* The lambda is : {}".format(lambda_for_ER))

    warmup_steps = 1e2

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

            b_input_ids1 = batch[0].to(device)
            b_labels1 = batch[0].to(device)
            b_masks1 = batch[1].to(device)

            b_input_ids2 = batch[2].to(device)
            b_labels2 = batch[2].to(device)
            b_masks2 = batch[3].to(device)

            model.zero_grad()        

            outputs = model(  b_input_ids1,
                              labels=b_labels1, 
                              attention_mask = b_masks1,
                              token_type_ids=None,
                              output_hidden_states=True
                            )

            flipped_outputs = model(  b_input_ids2,
                              labels=b_labels2, 
                              attention_mask = b_masks2,
                              token_type_ids=None,
                              output_hidden_states=True
                            )

            loss = get_loss(outputs, flipped_outputs, lambda_for_ER)
            loss = loss.mean()

            loss.backward()
            # loss.backward(torch.ones(gpu_count))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        logging.info("* Currrnt res is {}".format(ER_res))

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(dataloader)

        # Store the loss value for plotting the learning curve.
        ER_res['loss'].append(avg_train_loss)

    return model

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

    fine_tune_text_file = "../data/collected_sents/13436_flipped_gap_sents.pkl"
    with open(fine_tune_text_file, 'rb') as f:
        tune_data = pickle.load(f)
    logging.info("* {} sentences are used for finetuning".format(len(tune_data)))

    device = torch.device("cuda")

    args = parse_arguments()
    eval_type = args.eval_type
    batch_size = int(args.batch_size)
    lr = float(args.lr)
    epochs = int(args.epochs)
    lambda_for_ER = float(args.lambda_for_ER)
    pretrained_model = args.model

    model_size = ''
    if 'medium' in pretrained_model:
        model_size = 'medium'

    logging.info("* batch_size is {}".format(batch_size))

    spacy.load('en_core_web_lg')
    base_pairs = load_json_pairs('./CDA/gender_name_pairs/cda_default_pairs.json')
    name_pairs = load_json_pairs('./CDA/gender_name_pairs/names_pairs_1000_scaled.json')
    substitutor = Substitutor(base_pairs, name_pairs=name_pairs)

    ER_res = {}
    ER_res['loss'] = []
    ER_res['seat_ezs'] = []
    ER_res['ori_seat_ezs'] = []
    ER_res['IAs'] = []
    ER_res['err_nums'] = []

    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model, 
    #                                           bos_token='<|startoftext|>', 
    #                                           eos_token='<|endoftext|>', 
    #                                           pad_token='<|pad|>')

    dataset = ERDataset(tune_data, tokenizer, substitutor)

    dataloader = DataLoader(
            dataset,  
            sampler = RandomSampler(dataset), # Sampling for training is random
            batch_size = batch_size
        )

    model = GPT2LMHeadModel.from_pretrained(pretrained_model)
    # if torch.cuda.device_count() >= 1:
    #     logging.info("* Let's use {} GPUs!".format(torch.cuda.device_count()))
    #     model = torch.nn.DataParallel(model)

    st = time.time()

    model = fine_tune(model, dataloader, eval_type, tokenizer, lr, lambda_for_ER, device, ER_res, epochs = epochs)

    save_model(tokenizer, model, 'saved_models/CDS/CDS_gpt2_{}_{}_{}_{}_{}_{}'.format(model_size, eval_type, epochs, batch_size, lr, lambda_for_ER))

    # with open('large_res_for_opt_paras/ER/ER_gpt2_trade_off_{}_{}_{}_{}_{}.json'.format(eval_type, epochs, batch_size, lr, lambda_for_ER), 'w') as f:
    #     json.dump(ER_res, f, indent = '\t')