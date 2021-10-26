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
import spacy
from CDA.substitutor import Substitutor, load_json_pairs
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def save_model(tokenizer, model, file_name):
    tokenizer.save_pretrained(file_name)
    model.save_pretrained(file_name)
    print("saving model to {}".format(file_name))

    return   

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

def get_fine_tune_corpus(ori_file = '../data/13436_flipped_gap_sents.pkl', percent_words = 0.3):
    
    with open(ori_file, 'rb') as f:
        sents = pickle.load(f)
    
    encoder_sents = [noise_sentence(s, percent_words) + '</s>' for s in sents]
    
    sents = ['<s>' + s + '</s>'for s in sents]
    label_sents = [s[3:] for s in sents]
    decoder_sents = [s[:-4] for s in sents]
    
    return encoder_sents, decoder_sents, label_sents

class BartERDataset(Dataset):

    def __init__(self, encoder_sents, decoder_sents, label_sents, tokenizer, substitutor, max_length=128):
        self.tokenizer = tokenizer
        self.encoder_ids = []
        self.encoder_attn = []

        self.decoder_ids = []
        self.decoder_attn = []

        self.label_ids = []

        self.ori_sent_ids = []
        self.ori_sent_attn = []

        self.flipped_sent_ids = []
        self.flipped_sent_attn = []

        assert len(encoder_sents) == len(decoder_sents) == len(label_sents)

        ori_sents = [sent[:-4] for sent in label_sents]

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

            encodings_dict = tokenizer(ori_sents[i],
                                         truncation=True, 
                                         max_length=max_length, 
                                         padding="max_length")
            self.ori_sent_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.ori_sent_attn.append(torch.tensor(encodings_dict['attention_mask']))

            flipped_s = substitutor.invert_document(ori_sents[i])[0]
            encodings_dict = tokenizer(flipped_s,
                                         truncation=True, 
                                         max_length=max_length, 
                                         padding="max_length")
            self.flipped_sent_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.flipped_sent_attn.append(torch.tensor(encodings_dict['attention_mask']))
        
    def __len__(self):
        return len(self.encoder_ids)

    def __getitem__(self, idx):
        return self.encoder_ids[idx], self.encoder_attn[idx],\
               self.decoder_ids[idx], self.decoder_attn[idx],\
               self.label_ids[idx],\
               self.ori_sent_ids[idx], self.ori_sent_attn[idx],\
               self.flipped_sent_ids[idx], self.flipped_sent_attn[idx]

def get_loss(outputs, ori_emb, flipped_emb, lambda_for_ER):

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    cos_dis = cos(ori_emb, flipped_emb)
    # assert list(cos_dis.shape) == [1]

    loss = lambda_for_ER * (-cos_dis.mean()) + outputs.loss

    return loss

def fine_tune(model, dataloader, eval_type, tokenizer, lr, lambda_for_ER, device, ER_res, epochs = 3):
    model.to(device)
    model.train()

    logging.info("* The total epoch is : {}".format(str(epochs)))
    logging.info("* The learning rate is : {}".format(lr))
    logging.info("* The lambda is : {}".format(lambda_for_ER))

    warmup_steps = 1e2
    sample_every = 100

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

            model.train()
            model.zero_grad()        
            # mask inputs so the model can actually learn something
            b_encoder_ids = batch[0].to(device)
            b_encoder_attn = batch[1].to(device)

            b_decoder_ids = batch[2].to(device)
            b_decoder_attn = batch[3].to(device)

            b_label_ids = batch[4].to(device)

            outputs = model(input_ids=b_encoder_ids, attention_mask = b_encoder_attn,
                             decoder_input_ids=b_decoder_ids, decoder_attention_mask  = b_decoder_attn,
                             labels=b_label_ids)

            del b_encoder_ids
            del b_encoder_attn
            del b_decoder_ids
            del b_label_ids

            b_ori_sent_ids = batch[5].to(device)
            b_ori_sent_attn = batch[6].to(device)
            
            ori_emb = model(input_ids=b_ori_sent_ids, attention_mask = b_ori_sent_attn
                            ,output_hidden_states=True).decoder_hidden_states[-1]
            ori_emb = torch.mean(ori_emb, dim=1)

            del b_ori_sent_ids
            del b_ori_sent_attn

            b_flipped_sent_ids = batch[7].to(device)
            b_flipped_sent_attn = batch[8].to(device)

            flipped_emb = model(input_ids=b_flipped_sent_ids, attention_mask = b_flipped_sent_attn
                            ,output_hidden_states=True).decoder_hidden_states[-1]
            flipped_emb = torch.mean(flipped_emb, dim=1)

            del b_flipped_sent_ids
            del b_flipped_sent_attn

            loss = get_loss(outputs, ori_emb, flipped_emb, lambda_for_ER)
            loss = loss.mean()

            batch_loss = loss.item()
            total_loss += batch_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        logging.info("* Current res is {}".format(ER_res))

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(dataloader)
        ER_res['loss'].append(avg_train_loss)

    return model

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_type', required=True)
    parser.add_argument('--lambda_for_ER', required=True, default = 1.0)
    parser.add_argument('--epochs', required=True)
    parser.add_argument('--lr', required=True, default = 5e-4)
    parser.add_argument('--batch_size', required=True, default = 1)
    parser.add_argument('--model', help='which BERT model to use', required=False, default = 'bert-base-uncased')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    max_len = 128

    device = torch.device("cuda")

    encoder_sents, decoder_sents, label_sents = get_fine_tune_corpus()
    print("* {} sentences are used for finetuning".format(len(encoder_sents)))

    args = parse_arguments()
    eval_type = args.eval_type
    batch_size = int(args.batch_size)
    lr = float(args.lr)
    epochs = int(args.epochs)
    lambda_for_ER = float(args.lambda_for_ER)

    ER_res = {}
    ER_res['loss'] = []
    ER_res['seat_ezs'] = []
    ER_res['ori_seat_ezs'] = []
    ER_res['log_ezs'] = []
    ER_res['IAs'] = []
    ER_res['err_nums'] = []

    spacy.load('en_core_web_lg')
    base_pairs = load_json_pairs('./CDA/gender_name_pairs/cda_default_pairs.json')
    name_pairs = load_json_pairs('./CDA/gender_name_pairs/names_pairs_1000_scaled.json')
    substitutor = Substitutor(base_pairs, name_pairs=name_pairs)

    pretrained_model = args.model
    tokenizer = BartTokenizer.from_pretrained(pretrained_model)

    model_size = 'base'
    if 'large' in pretrained_model:
        model_size = 'large'

    dataset = BartERDataset(encoder_sents, decoder_sents, label_sents, tokenizer, substitutor, max_length = max_len)

    logging.info("* batch_size is {}".format(batch_size))
    dataloader = DataLoader(
            dataset,  
            sampler = RandomSampler(dataset), # Sampling for training is random
            batch_size = batch_size
        )

    model = BartForConditionalGeneration.from_pretrained(pretrained_model,
                                                output_attentions=False,
                                                output_hidden_states=False)
    # if torch.cuda.device_count() >= 1:
    #     logging.info("* Let's use {} GPUs!".format(torch.cuda.device_count()))
    #     model = torch.nn.DataParallel(model)

    st = time.time()

    print('-- Set up model fine-tuning --')
    model = fine_tune(model, dataloader, eval_type, tokenizer, lr, lambda_for_ER, device, ER_res, epochs = epochs)

    et = time.time()
    logging.info('ER took {0:.2f} minutes'.format((et - st) / 60))

    save_model(tokenizer, model, 'saved_models/ER/ER_{}_bart_{}_{}_{}_{}_{}'.format(model_size, eval_type, epochs, batch_size, lr, lambda_for_ER))

    # with open('saved_models/ER/ER_{}_bart_{}_{}_{}_{}_{}.pkl'.format(model_size, eval_type, epochs, batch_size, lr, lambda_for_ER), 'wb') as f:
    #     pickle.dump(model, f)