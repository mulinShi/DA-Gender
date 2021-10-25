import sys 
from sklearn.decomposition import PCA
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
import re

import json
import datetime
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_type', required=True, default = 50)
    parser.add_argument('--target_pair_ratio', required=True)
    parser.add_argument('--model', help='which BERT model to use', required=False, default = 'bert-base-uncased')
    parser.add_argument('--output_model', help='output_model', required=False, default = 'test_trade_off/CDS_13436_flipped_gap')
    parser.add_argument('--text_type', help='output_model', required=False, default = 'gap')
    args = parser.parse_args()
    return args

def load_json_pairs(input_file):
    with open(input_file, "r") as fp:
        pairs = json.load(fp)
    return pairs

def load_define_pairs(args, target_pair_ratio):
    define_pairs = []
    if args.text_type == 'gap':
        define_pairs = []   # (female, male)
        base_pairs = load_json_pairs('./CDA/gender_name_pairs/cda_default_pairs.json')
        name_pairs = load_json_pairs('./CDA/gender_name_pairs/names_pairs_1000_scaled.json')
        addtitional_pairs = [["guy", "gal"], ["his", "her"], ["himself", "herself"], ["guys", "gals"]]
        for pair in base_pairs + name_pairs + addtitional_pairs:
            define_pairs.append([pair[1], pair[0]])
    elif args.text_type == 'wiki':
        define_pairs = [["woman", "man"], ["girl", "boy"], ["she", "he"], ["mother", "father"], ["daughter", "son"], ["gal", "guy"], ["female", "male"], ["her", "his"], ["herself", "himself"], ["Mary", "John"]]
    else:
        raise ValueError('The agrs.text_type is incorrect : {}'.format(args.text_type))

    size = int(len(define_pairs) * target_pair_ratio)
    logging.info("* the size of the total pair set is {}, the ratio is {}, the size of used pair set is {}".\
        format(len(define_pairs), target_pair_ratio, size))

    define_pairs = define_pairs[:size]

    return define_pairs

def match(a,L):
    for b in L:
        if a == b:
            return True
    return False

def replace(a,new,L):
    Lnew = []
    for b in L:
        if a == b:
            Lnew.append(new)
        else:
            Lnew.append(b)
    return ' '.join(Lnew)

def get_def_sent_pairs(all_sents_file_names, define_pairs):
    sent_pairs = []
    with open(all_sents_file_names, 'rb') as f:
        all_sents = pickle.load(f)
    for sent in all_sents:
        ws = sent.split(" ")
        for i, (female, male) in enumerate(define_pairs):
            if match(female, ws):
                sent_f = sent
                sent_m = replace(female,male, ws)
                sent_pairs.append((sent_f, sent_m))
                # sent_pairs[i]['f'].append(sent_f)
                # sent_pairs[i]['m'].append(sent_m)
            if match(male, ws):
                sent_f = replace(male,female, ws)
                sent_m = sent
                sent_pairs.append((sent_f, sent_m))
                # sent_pairs[i]['f'].append(sent_f)
                # sent_pairs[i]['m'].append(sent_m)
    logging.info("* {} sent defining pairs are found".format(len(sent_pairs)))
    return sent_pairs

def doPCA(matrix, num_components=10):
    print("* gender dir dim is", num_components)
    pca = PCA(n_components=num_components, svd_solver="auto")
    pca.fit(matrix) # Produce different results each time...
    return pca

def get_embedding_dic(define_pairs, tokenizer, model, device, args):

    if args.text_type == 'gap':
        all_sents_file_names = '../data/13436_flipped_gap_sents.pkl'
    elif args.text_type == 'wiki':
        all_sents_file_names = '../data/collected_sents/wiki.pkl'
    else:
        raise ValueError('The agrs.text_type is incorrect : {}'.format(args.text_type))

    sent_pairs = get_def_sent_pairs(all_sents_file_names, define_pairs)

    all_sents = [t[0] for t in sent_pairs] + [t[1] for t in sent_pairs]

    # embs = get_only_embeddings(all_sents, tokenizer, model, device, max_len)
    embs = []
    for sent in all_sents:
        encodings_dict = tokenizer(sent)
        ids = torch.tensor(encodings_dict['input_ids'])
        attms = torch.tensor(encodings_dict['attention_mask'])

        emb = model(input_ids = ids.reshape([-1, ids.shape[0]])
                , attention_mask = attms.reshape([-1, attms.shape[0]]), output_hidden_states=True).decoder_hidden_states[-1]
        emb = torch.mean(emb, dim=1).cpu().detach().numpy()
        embs.append(emb)

    sent2emb = {}
    for i, sent in enumerate(all_sents):
        sent2emb[sent] = embs[i]
        # sent2emb[sent] = np.reshape(embs[i], (-1, embs[i].shape[0]))

    logging.info("******The length of sent2emb1 is {}".format(len(sent2emb)))
     
    return sent_pairs, sent2emb

def evaluate_by_dif_data(sent_pairs, sent2emb_eval, sent_res, device, eval_type, k = 1, keepdims=True):

    used_pairs = sent_pairs
    # logging.info("The shape for sent2emb[t[]]")
    all_embs_f = [sent2emb_eval[t[0]] for t in used_pairs]
    all_embs_m = [sent2emb_eval[t[1]] for t in used_pairs]

    all_embs_f = np.concatenate(all_embs_f, axis=0)
    # logging.info("****** The shape of all_embs_f/all_embs_m is {}".format(all_embs_f.shape))
    all_embs_f /= np.linalg.norm(all_embs_f, axis=-1, keepdims=True)
    all_embs_m = np.concatenate(all_embs_m, axis=0)
    all_embs_m /= np.linalg.norm(all_embs_m, axis=-1, keepdims=True)
    # logging.info("****** The shape of normalized all_embs_f/all_embs_m is {}".format(all_embs_f.shape))        

    means = (all_embs_f + all_embs_m) / 2.0
    all_embs_f -= means
    all_embs_m -= means
    all_embeddings = np.concatenate([all_embs_f, all_embs_m], axis=0)

    gender_dir = doPCA(all_embeddings, num_components = 10).components_[:k]
    if (not keepdims):
        gender_dir = np.mean(gender_dir, axis=0)

    # assert gender_dir.shape == (1, 1024)
    return gender_dir

if __name__ == '__main__':

    args = parse_arguments()

    eval_type = args.eval_type

    target_pair_ratio = float(args.target_pair_ratio)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model_size = 'base'
    if 'large' in args.model:
        model_size = 'large'

    tokenizer = BartTokenizer.from_pretrained(args.model)
    model = BartForConditionalGeneration.from_pretrained(args.model,
                                            output_attentions=False,
                                            output_hidden_states=False)

    define_pairs = load_define_pairs(args, target_pair_ratio)
    sent_pairs, sent2emb = get_embedding_dic(define_pairs, tokenizer, model, device, args)

    sent_res = {}

    gender_dir = evaluate_by_dif_data(sent_pairs, sent2emb, sent_res, device, eval_type)

    with open('saved_models/sent_debias/SENT_debias_{}_bart_{}_{}.pkl'.format(model_size, eval_type, target_pair_ratio), 'wb') as f:
        pickle.dump(gender_dir, f)
