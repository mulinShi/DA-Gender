import sys
sys.path.append("..") 
import pickle
import json
import numpy as np
from transformers import BartTokenizer, BartForConditionalGeneration
from bias_utils.utils_for_diverse_tmps import get_only_embeddings
import torch
import math
import argparse
import logging
# from weat import *

def exact_mc_perm_test(xs, ys, nmc=100000):
    n, k = len(xs), 0
    diff = np.abs(np.mean(xs) - np.mean(ys))
    zs = np.concatenate([xs, ys])
    for j in range(nmc):
#         print(j)
        np.random.shuffle(zs)
        k += diff < np.abs(np.mean(zs[:n]) - np.mean(zs[n:]))
    return k / nmc

def dropspace(u, V):
    norm_sqrd = np.sum(V*V, axis=-1)
    vecs = np.divide(V@u, norm_sqrd)[:, None] * V
    subspace = np.sum(vecs, axis=0)
    return u - subspace

def cossim(x, y):
    return np.dot(x, y) / math.sqrt(np.dot(x, x) * np.dot(y, y))

def get_embeddings(corpus, model_type = None, subspace_path = None, tokenizer = None, device = None):
    
    gender_dir = None
    if subspace_path is not None:
        if type(subspace_path) is str:
            dir_path = "{}".format(subspace_path)
            print("* load gender dir from ", dir_path)
            with open(dir_path, 'rb') as f:
                gender_dir = pickle.load(f)
        else:
            print("* The subspace_path is already input")
            gender_dir = subspace_path

    all_sents_list = corpus['targ1'] + corpus['targ2']\
            + corpus['attr1'] + corpus['attr2']
    all_sents_list = list(set(all_sents_list))
    print("--- {} sentens to be processes ---".format(len(all_sents_list)))

    sent2emb = {}

    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print('We will use the GPU:', torch.cuda.get_device_name(0))
        else:
            print('No GPU available, using the CPU instead.')
            device = torch.device('cpu')
         
    if type(model_type) is str:
        pretrained_model = model_type
        tokenizer = BartTokenizer.from_pretrained(pretrained_model)
        model = BartForConditionalGeneration.from_pretrained(pretrained_model)
    else:
        print("* The mode is already input.")
        model = model_type

    print("Testing")
    all_embeddings = get_only_embeddings(all_sents_list, tokenizer, model, device, 128)
    if gender_dir is not None:
        print("debiasing by sent_debias...")
        for i, emb in enumerate(all_embeddings):
            emb /= np.linalg.norm(emb)
            emb = dropspace(emb, gender_dir)
            emb /= np.linalg.norm(emb)
            all_embeddings[i] = emb
    for i, _ in enumerate(all_sents_list):
        sent2emb[all_sents_list[i]] = all_embeddings[i]
    # with open('sent2emb.pkl', 'wb') as f:
    #     pickle.dump(sent2emb, f)
    return sent2emb

def compute_effect_size_by_stardard_seat(corpus, sent2emb = None):
    X = {sent : sent2emb[sent] for sent in corpus['targ1']}
    Y = {sent : sent2emb[sent] for sent in corpus['targ2']}
    A = {sent : sent2emb[sent] for sent in corpus['attr1']}
    B = {sent : sent2emb[sent] for sent in corpus['attr2']}

    (X, Y) = convert_keys_to_ints(X, Y)
    (A, B) = convert_keys_to_ints(A, B)

    XY = X.copy()
    XY.update(Y)
    AB = A.copy()
    AB.update(B)

    cossims = construct_cossim_lookup(XY, AB)
    logger.info("computing pval...")
    pval = p_val_permutation_test(X, Y, A, B, cossims=cossims, n_samples=10000)
    logger.info("pval: %g", pval)

    logger.info("computing effect size...")
    esize = effect_size(X, Y, A, B, cossims=cossims)
    logger.info("esize: %g", esize)

    # return seat_effect_size, seat_p

def compute_effect_size(corpus, sent2emb = None):

    targ1 = corpus['targ1']
    targ2 = corpus['targ2']
    attr1 = corpus['attr1']
    attr2 = corpus['attr2']

    asso1 = []
    asso2 = []

    for t in targ1:
        asso = np.array([cossim(sent2emb[t], sent2emb[a]) for a in attr1]).mean()\
               - np.array([cossim(sent2emb[t], sent2emb[a]) for a in attr2]).mean()
        asso1.append(asso)
    for t in targ2:
        asso = np.array([cossim(sent2emb[t], sent2emb[a]) for a in attr1]).mean()\
               - np.array([cossim(sent2emb[t], sent2emb[a]) for a in attr2]).mean()
        asso2.append(asso)

    asso1 = np.array(asso1)
    asso2 = np.array(asso2)
    
    diff = (asso1.mean() - asso2.mean())
    std_ = np.concatenate([asso1, asso2], axis=0).std() + 1e-8
#     print(np.concatenate([asso1, asso2], axis=0).shape)
    seat_effect_size = diff / std_
#     print(diff, ' ', std_)
    seat_p = exact_mc_perm_test(asso1, asso2)
    # print("for seat effect size is {} p value is {}".format(effect_size, p))

    return seat_effect_size, seat_p

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', help='which model to use', required=False, default = None)
    parser.add_argument('--subspace_path', help='', required=False, default = None)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    file = 'ori_seat_data.json'
    # file = '../data/ori_seat_data.json'
    with open(file, 'r') as f:
        test = json.load(f)
    
    args = parse_arguments()
    model_type = args.model_type
    subspace_path = args.subspace_path

    sent2emb = get_embeddings(test, model_type = model_type, subspace_path = subspace_path)

    # compute_effect_size_by_stardard_seat(test, sent2emb = sent2emb)

    seat_ez, p = compute_effect_size(test, sent2emb = sent2emb)
    print("seat effect size is {}, p value is {}".format(seat_ez, p))
