import sys 
import argparse
import math
import random
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset, RandomSampler, DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
import pickle
import datetime
import json
import logging
import re
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import spacy
from CDA.substitutor import Substitutor, load_json_pairs


def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

def noise_sentence(sentence_, percent_words, replacement_token = "<mask>"):
    '''
    Function that noises a sentence by adding <mask> tokens
    Args: sentence - the sentence to noise
        percent_words - the percent of words to replace with <mask> tokens; the number is rounded up using math.ceil
    Returns a noised sentence
    '''
    # Create a list item and copy
    sentence_ = sentence_.split(' ')
    sentence = sentence_.copy()

    num_words = math.ceil(len(sentence) * percent_words)

    # Create an array of tokens to sample from; don't include the last word as an option because in the case of lyrics
    # that word is often a rhyming word and plays an important role in song construction
    sample_tokens = set(np.arange(0, np.maximum(1, len(sentence)-1)))

    words_to_noise = random.sample(sample_tokens, num_words)

    # Swap out words, but not full stops
    for pos in words_to_noise:
        if sentence[pos] != '.':
            sentence[pos] = replacement_token

    # Remove redundant spaces
    sentence = re.sub(r' {2,5}', ' ', ' '.join(sentence))

    # Combine concurrent <mask> tokens into a single token; this just does two rounds of this; more could be done
    sentence = re.sub(r'<mask> <mask>', "<mask>", sentence)
    sentence = re.sub(r'<mask> <mask>', "<mask>", sentence)
    return sentence

def get_fine_tune_corpus(CDS_ratio, percent_words = 0.3):
    
    ############

    spacy.load('en_core_web_lg')
    base_pairs = load_json_pairs('./CDA/gender_name_pairs/cda_default_pairs.json')
    name_pairs = load_json_pairs('./CDA/gender_name_pairs/names_pairs_1000_scaled.json')
    substitutor = Substitutor(base_pairs, name_pairs=name_pairs)

    file_name = "../data/13436_ori_gap_sents.pkl"
    with open(file_name, 'rb') as f:
        sents = pickle.load(f)
    logging.info("* {} sentences are used for finetuning".format(len(sents)))

    size = int(CDS_ratio * len(sents))
    logging.info("* CDS_ratio : {}, the number of correponding inverted sentences {}".format(CDS_ratio, size))

    sents_A = sents[:size]
    sents_A = [substitutor.invert_document(sent)[0] for sent in sents_A]
    sents_B = sents[size:]

    assert len(sents_A + sents_B) == len(sents)

    sents = sents_A + sents_B

    ############
    
    encoder_sents = [noise_sentence(s, percent_words) + '</s>' for s in sents]
    
    sents = ['<s>' + s + '</s>'for s in sents]
    label_sents = [s[3:] for s in sents]
    decoder_sents = [s[:-4] for s in sents]
    
    return encoder_sents, decoder_sents, label_sents

class BartDataset(Dataset):

    def __init__(self, encoder_sents, decoder_sents, label_sents, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.encoder_ids = []
        self.encoder_attn = []

        self.decoder_ids = []
        self.decoder_attn = []

        self.label_ids = []

        assert len(encoder_sents) == len(decoder_sents) == len(label_sents)

        for i in range(len(encoder_sents)):
            encodings_dict = tokenizer(encoder_sents[i],
                                         add_special_tokens=False,
                                         truncation=True, 
                                         max_length=max_length, 
                                         padding="max_length")
            self.encoder_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.encoder_attn.append(torch.tensor(encodings_dict['attention_mask']))

            encodings_dict = tokenizer(decoder_sents[i],
                                         add_special_tokens=False,
                                         truncation=True, 
                                         max_length=max_length, 
                                         padding="max_length")
            self.decoder_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.decoder_attn.append(torch.tensor(encodings_dict['attention_mask']))

            encodings_dict = tokenizer(label_sents[i],
                                         add_special_tokens=False,
                                         truncation=True, 
                                         max_length=max_length, 
                                         padding="max_length")
            self.label_ids.append(torch.tensor(encodings_dict['input_ids']))
        
    def __len__(self):
        return len(self.encoder_ids)

    def __getitem__(self, idx):
        return self.encoder_ids[idx], self.encoder_attn[idx], self.decoder_ids[idx], self.decoder_attn[idx], self.label_ids[idx] 

def fine_tune(model, dataloader, tokenizer, device, CDS_res, lr, eval_type, epochs = 3):
    model.to(device)
    model.train()

    warmup_steps = 1e2
    sample_every = 100

    logging.info("* The total epoch is : {}".format(str(epochs)))
    logging.info("* learning rate is {}".format(lr))
    optimizer = AdamW(model.parameters(),
                      lr = lr,
                      eps = 1e-8
                    )

    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = warmup_steps, 
                                                num_training_steps = total_steps)

    for epoch_i in range(0, epochs):
        logging.info("* The current epoch is : {}".format(str(epoch_i)))
        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0

        for step, batch in enumerate(dataloader):

            # mask inputs so the model can actually learn something
            b_encoder_ids = batch[0].to(device)
            b_encoder_attn = batch[1].to(device)

            b_decoder_ids = batch[2].to(device)
            b_decoder_attn = batch[3].to(device)

            b_label_ids = batch[4].to(device)

            model.zero_grad()        

            outputs = model(input_ids=b_encoder_ids, attention_mask = b_encoder_attn,
                             decoder_input_ids=b_decoder_ids, decoder_attention_mask  = b_decoder_attn,
                             labels=b_label_ids)

            loss = outputs[0]  

            batch_loss = loss.item()
            total_loss += batch_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(dataloader)
        CDS_res['loss'].append(avg_train_loss)

        logging.info("* Current res is {}".format(CDS_res))

    return model

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--CDS_ratio', required=True)
    parser.add_argument('--eval_type', required=True)
    parser.add_argument('--lr', required=True, default = 5e-4)
    parser.add_argument('--batch_size', required=True, default = 1)
    parser.add_argument('--epochs', required=True)
    parser.add_argument('--model', help='which BERT model to use', required=False, default = 'bert-base-uncased')
    # parser.add_argument('--model_size', required=True)
    args = parser.parse_args()
    return args

def save_model(tokenizer, model, file_name):
    tokenizer.save_pretrained(file_name)
    model.save_pretrained(file_name)
    print("saving model to {}".format(file_name))

    return   

if __name__ == '__main__':

    max_len = 128

    args = parse_arguments()
    CDS_ratio = float(args.CDS_ratio)
    eval_type = args.eval_type
    batch_size = int(args.batch_size)
    lr = float(args.lr)
    epochs = int(args.epochs)
    logging.info("* batch_size is {}".format(batch_size))

    pretrained_model = args.model
    logging.info("* used original model is {}".format(pretrained_model))

    model_size = 'base'
    if 'large' in pretrained_model:
        model_size = 'large'

    encoder_sents, decoder_sents, label_sents = get_fine_tune_corpus(CDS_ratio)
    print("* {} sentences are used for finetuning".format(len(encoder_sents)))

    CDS_res = {}
    CDS_res['loss'] = []
    CDS_res['seat_ezs'] = []
    CDS_res['ori_seat_ezs'] = []
    CDS_res['log_ezs'] = []
    CDS_res['IAs'] = []
    CDS_res['err_nums'] = []

    tokenizer = BartTokenizer.from_pretrained(pretrained_model)

    dataset = BartDataset(encoder_sents, decoder_sents, label_sents, tokenizer, max_length = max_len)

    dataloader = DataLoader(
            dataset,  
            sampler = RandomSampler(dataset), # Sampling for training is random
            batch_size = batch_size
        )

    model = BartForConditionalGeneration.from_pretrained(pretrained_model,
                                                output_attentions=False,
                                                output_hidden_states=False)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    st = time.time()

    print('-- Set up model fine-tuning --')
    model = fine_tune(model, dataloader, tokenizer, device, CDS_res, lr, eval_type, epochs = epochs)

    et = time.time()
    logging.info('CDS took {0:.2f} minutes'.format((et - st) / 60))

    save_model(tokenizer, model, 'saved_models/CDS/CDS_{}_bart_{}_{}_{}_{}_{}'.format(model_size, CDS_ratio, eval_type, epochs, batch_size, args.lr))

    # with open('saved_models/CDS/CDS_{}_bart_{}_{}_{}_{}_{}.pkl'.format(model_size, CDS_ratio, eval_type, epochs, batch_size, args.lr), 'wr') as f:
    #     pickle.dump(model, f)