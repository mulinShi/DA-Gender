import pandas as pd
import json
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, BartTokenizer, BartForConditionalGeneration, BartForCausalLM, BertTokenizer, BertForMaskedLM, AdamW, get_linear_schedule_with_warmup
import pickle
import math
import sys
# sys.path.append( '../../' )
from tqdm import tqdm, trange
from torch.nn.functional import softmax
import transformers
import os
import argparse

def get_vios_and_dif(data):
    violates = []
    difs = []
    for i, d in enumerate(data.Sent_A):
        male_asso = data.pref_to_male[i]
        female_asso = data.pref_to_female[i]
        if data.ideal_pref[i] == 'male':
            dif = male_asso - female_asso
        else:
            dif = female_asso - male_asso
        difs.append(dif)
        if dif <= 0:
            violates.append(i)
    return violates, difs

# def get_embs_for_autoregressive_models(sents, tokenizer, model, device):

#     model.to(device)
#     model.eval()

#     all_embeddings = []
#     for i, sent in enumerate(tqdm(sents)):
#         tokens = tokenizer.encode(sents[i])
#         tokens_tensor = torch.tensor(
#                     tokens).to(device).unsqueeze(0)
#         with torch.no_grad():
#             embs = model(tokens_tensor, output_hidden_states=True).hidden_states[-1][:, -1, :].cpu().detach().numpy()
#         assert list(embs.shape) == [1, 768]
#         all_embeddings.append(embs)
#     all_embeddings = np.concatenate(all_embeddings, axis=0)
#     return all_embeddings

def dropspace(u, V):
    norm_sqrd = np.sum(V*V, axis=-1)
    vecs = np.divide(V@u, norm_sqrd)[:, None] * V
    subspace = np.sum(vecs, axis=0)
    return u - subspace

def get_asso_bart(Sent_As, Sent_Bs, ideal_prefs, tokenizer, model, device, model_type, gender_dir):

    model.to(device)
    model.eval()

    pref2males = []
    pref2females = []
    for i, _ in enumerate(tqdm(Sent_As)):

        tokenAs = tokenizer.encode(Sent_As[i])
        tokens_tensorA = torch.tensor(
                    tokenAs).to(device).unsqueeze(0)

        tokenBs = tokenizer.encode(Sent_Bs[i])
        tokens_tensorB = torch.tensor(
                    tokenBs).to(device).unsqueeze(0)

        with torch.no_grad():
            joint_sentence_probability = []
            if gender_dir is not None:
                output_before_transformer = model.transformer(input_ids = tokens_tensorA).last_hidden_state.cpu().detach().numpy()
                for b in range(output_before_transformer.shape[0]):
                    for i in range(output_before_transformer.shape[1]):
                        emb = output_before_transformer[b, i, :]
                        emb /= np.linalg.norm(emb)
                        emb = dropspace(emb, gender_dir)
                        emb /= np.linalg.norm(emb)
                        output_before_transformer[b, i, :] = emb
                output_before_transformer = torch.tensor(output_before_transformer).to(device)
                outputs = model.lm_head(output_before_transformer)
                output = torch.softmax(outputs, dim=-1)
            else:
                output = torch.softmax(model(input_ids = tokens_tensorA)[0], dim=-1)
            for idx in range(len(tokenAs) - 2):
                joint_sentence_probability.append(
                    output[0, idx, tokenAs[idx + 1]].item())
            assert (len(tokenAs) - 2) == len(joint_sentence_probability)
            score = np.sum([np.log2(i) for i in joint_sentence_probability]) 
            score /= len(joint_sentence_probability)
            if ideal_prefs[i] == 'male':
                pref2males.append(np.power(2, score))
            else:
                assert ideal_prefs[i] == 'female'
                pref2females.append(np.power(2, score))

            joint_sentence_probability = []
            if gender_dir is not None:
                output_before_transformer = model.transformer(input_ids = tokens_tensorB).last_hidden_state.cpu().detach().numpy()
                for b in range(output_before_transformer.shape[0]):
                    for i in range(output_before_transformer.shape[1]):
                        emb = output_before_transformer[b, i, :]
                        emb /= np.linalg.norm(emb)
                        emb = dropspace(emb, gender_dir)
                        emb /= np.linalg.norm(emb)
                        output_before_transformer[b, i, :] = emb
                output_before_transformer = torch.tensor(output_before_transformer).to(device)
                outputs = model.lm_head(output_before_transformer)
                output = torch.softmax(outputs, dim=-1)
            else:
                output = torch.softmax(model(input_ids = tokens_tensorB)[0], dim=-1)
            for idx in range(len(tokenAs) - 2):
                joint_sentence_probability.append(
                    output[0, idx, tokenBs[idx + 1]].item())
            assert (len(tokenBs) - 2) == len(joint_sentence_probability)
            score = np.sum([np.log2(i) for i in joint_sentence_probability]) 
            score /= len(joint_sentence_probability)
            if ideal_prefs[i] == 'male':
                pref2females.append(np.power(2, score))
            else:
                assert ideal_prefs[i] == 'female'
                pref2males.append(np.power(2, score))

    return pref2males, pref2females

def get_asso_gpt2(Sent_As, Sent_Bs, ideal_prefs, tokenizer, model, device, model_type, gender_dir):

    model.to(device)
    model.eval()

    pref2males = []
    pref2females = []
    for i, _ in enumerate(tqdm(Sent_As)):
        start_token = tokenizer.encode(tokenizer.bos_token)
        # start_token = []

        tokenAs = start_token + tokenizer.encode(Sent_As[i])
        tokens_tensorA = torch.tensor(
                    tokenAs).to(device).unsqueeze(0)

        tokenBs = start_token + tokenizer.encode(Sent_Bs[i])
        tokens_tensorB = torch.tensor(
                    tokenBs).to(device).unsqueeze(0)

        with torch.no_grad():
            joint_sentence_probability = []
            if gender_dir is not None:
                # print("* Using gender dir")
                output_before_transformer = model.transformer(input_ids = tokens_tensorA).last_hidden_state.cpu().detach().numpy()
                for b in range(output_before_transformer.shape[0]):
                    for i in range(output_before_transformer.shape[1]):
                        emb = output_before_transformer[b, i, :]
                        emb /= np.linalg.norm(emb)
                        emb = dropspace(emb, gender_dir)
                        emb /= np.linalg.norm(emb)
                        output_before_transformer[b, i, :] = emb
                output_before_transformer = torch.tensor(output_before_transformer).to(device)
                outputs = model.lm_head(output_before_transformer)
                output = torch.softmax(outputs, dim=-1)
            else:
                output = torch.softmax(model(input_ids = tokens_tensorA)[0], dim=-1)
            for idx in range(1, len(tokenAs)):
                joint_sentence_probability.append(
                    output[0, idx-1, tokenAs[idx]].item())
            assert (len(tokenAs) - 1) == len(joint_sentence_probability)
            score = np.sum([np.log2(i) for i in joint_sentence_probability]) 
            score /= len(joint_sentence_probability)
            if ideal_prefs[i] == 'male':
                pref2males.append(np.power(2, score))
            else:
                assert ideal_prefs[i] == 'female'
                pref2females.append(np.power(2, score))

            joint_sentence_probability = []
            if gender_dir is not None:
                output_before_transformer = model.transformer(input_ids = tokens_tensorB).last_hidden_state.cpu().detach().numpy()
                for b in range(output_before_transformer.shape[0]):
                    for i in range(output_before_transformer.shape[1]):
                        emb = output_before_transformer[b, i, :]
                        emb /= np.linalg.norm(emb)
                        emb = dropspace(emb, gender_dir)
                        emb /= np.linalg.norm(emb)
                        output_before_transformer[b, i, :] = emb
                output_before_transformer = torch.tensor(output_before_transformer).to(device)
                outputs = model.lm_head(output_before_transformer)
                output = torch.softmax(outputs, dim=-1)
            else:
                output = torch.softmax(model(input_ids = tokens_tensorB)[0], dim=-1)
            for idx in range(1, len(tokenBs)):
                joint_sentence_probability.append(
                    output[0, idx-1, tokenBs[idx]].item())
            assert (len(tokenBs) - 1) == len(joint_sentence_probability)
            score = np.sum([np.log2(i) for i in joint_sentence_probability]) 
            score /= len(joint_sentence_probability)
            if ideal_prefs[i] == 'male':
                pref2females.append(np.power(2, score))
            else:
                assert ideal_prefs[i] == 'female'
                pref2males.append(np.power(2, score))

    return pref2males, pref2females


def add_predicts(corpus, model_type = None, device = None, tokenizer = None, subspace_path = None):
    
    gender_dir = None
    if subspace_path is not None:
        if type(subspace_path) == str:
            dir_path = "{}".format(subspace_path)
            print("* load gender dir from ", dir_path)
            with open(dir_path, 'rb') as f:
                gender_dir = pickle.load(f)
        else:
            print("* The input gender dir is used.")
            gender_dir = subspace_path

    Sent_As = [i for i in corpus.Sent_A]
    Sent_Bs = [i for i in corpus.Sent_B]
    ideal_prefs = [i for i in corpus.ideal_pref]

    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    # if tokenizer is None:
    #     tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        
    # if type(model_type) is str:
    #     model = GPT2LMHeadModel.from_pretrained("gpt2") 
    # else:
    #     model = model_type

    if type(model_type) == str:
        print("Loading the model and the tokenizer from {}".format(model_type))
        model = GPT2LMHeadModel.from_pretrained(model_type)
        tokenizer = GPT2Tokenizer.from_pretrained(model_type)
    else:
        print("The model is already input")
        model = model_type
        if tokenizer is None:
            tokenizer = GPT2Tokenizer.from_pretrained("medium")

    pref2males, pref2females = get_asso_gpt2(Sent_As, Sent_Bs, ideal_prefs,\
                                       tokenizer, model, device, model_type, gender_dir)

    corpus = corpus.assign(pref_to_male = pref2males)
    corpus = corpus.assign(pref_to_female = pref2females)

    corpus.to_csv('sent_debias_{}_{}_over_debiasing.tsv'.format(len(corpus), model_type), sep='\t')

    return corpus

def parse_arguments():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--repeat_times', required=True, default = 1)
    parser.add_argument('--model_path', help='which model to use', required=False, default = 'facebook/bart-base')
    parser.add_argument('--model_type', help='which model to use', required=False, default = 'bart')
    parser.add_argument('--debiasing_type', help='', required=False, default = 'None')
    parser.add_argument('--subspace_path', help='', required=False, default = None)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    transformers.logging.ERROR

    args = parse_arguments()
    model_path = args.model_path
    model_type = args.model_type
    debiasing_type = args.debiasing_type
    subspace_path = args.subspace_path

    assert model_type == 'gpt2' or model_type == 'bart'

    data_without_preds_path = 'data/2610_sents_for_evaluating_gender_loss.tsv'

    over_debias = {}

    data_without_preds = pd.read_csv(data_without_preds_path, sep='\t', index_col = 0)

    data = add_predicts(data_without_preds, model_type = model_type, model_path = model_path, subspace_path = subspace_path)
    violates, difs = get_vios_and_dif(data)

    error_num = len(violates)
    over_debias['error_num'] = error_num

    error_rate = len(violates)/len(data)
    over_debias['error_rate'] = error_rate

    IA = sum(difs) / len(difs)
    over_debias['IA'] = IA

    with open('res/{}_{}_over_debiasing.json'.format(debiasing_type, model_type), 'w') as f:
        json.dump(over_debias, f, indent = 6)
