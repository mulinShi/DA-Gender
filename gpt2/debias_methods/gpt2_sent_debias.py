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
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
import pickle
from debias_utils import CDS_input_pipeline, format_time

# from evaluation_for_trade_off.evaluating_over_debiasing_gpt2 import add_predicts, get_vios_and_dif
# from evaluation_for_trade_off.evaluate_by_divers_tmps_gpt_2 import convert_tmps_to_sents, comput_asso_for_autoregressive, add_assos
# from evaluation_for_trade_off.gpt2_ori_seat import get_embeddings, compute_effect_size

import json
import datetime
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_type', required = True)
    parser.add_argument('--target_pair_ratio', required = True)
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
    logging.info("** gender dir dim (the number of components) is : {}".format(num_components))
    pca = PCA(n_components=num_components, svd_solver="auto")
    pca.fit(matrix) # Produce different results each time...
    return pca

def get_embedding_dic(define_pairs, tokenizer, model, device, args):

    # max_len = 128

    if args.text_type == 'gap':
        all_sents_file_names = '../data/collected_sents/13436_flipped_gap_sents.pkl'
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
        emb = model(input_ids = torch.tensor(encodings_dict['input_ids'])
                , attention_mask=torch.tensor(encodings_dict['attention_mask']), output_hidden_states=True).hidden_states[-1][-1, :].cpu().detach().numpy()
        embs.append(emb)

    sent2emb = {}
    for i, sent in enumerate(all_sents):
        sent2emb[sent] = np.reshape(embs[i], (-1, embs[i].shape[0]))

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

    logging.info("* Current res is {}".format(sent_res))

    return gender_dir

if __name__ == '__main__':

    args = parse_arguments()
    eval_type = args.eval_type
    target_pair_ratio = float(args.target_pair_ratio)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model_size = ''
    if 'medium' in args.model:
        model_size = 'medium'

    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    model = GPT2LMHeadModel.from_pretrained(args.model,
                                            output_attentions=False,
                                            output_hidden_states=False)

    define_pairs = load_define_pairs(args, target_pair_ratio)
    sent_pairs, sent2emb = get_embedding_dic(define_pairs, tokenizer, model, device, args)

    sent_res = {}
    sent_res['loss'] = []
    sent_res['seat_ezs'] = []
    sent_res['ori_seat_ezs'] = []
    sent_res['IAs'] = []
    sent_res['err_nums'] = []

    gender_dir = evaluate_by_dif_data(sent_pairs, sent2emb, sent_res, device, eval_type)

    # logging.info("* sent_res: ".format(sent_res))

    with open('saved_models/sent_debias/SENT_debias_gpt2_{}_{}_{}.pkl'.format(model_size, eval_type, target_pair_ratio), 'wb') as f:
        pickle.dump(gender_dir, f)

    # with open('large_res_for_opt_paras/sent/SENT_gpt2_trade_off_{}_{}.json'.format(eval_type, target_pair_ratio), 'w') as f:
    #     json.dump(sent_res, f, indent = '\t')
