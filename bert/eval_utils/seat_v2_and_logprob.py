import pandas as pd
import json
import numpy as np
import torch
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from transformers import BertTokenizer, BertForMaskedLM, AdamW, get_linear_schedule_with_warmup

from bias_utils.utils_for_diverse_tmps import model_evaluation_by_seat, model_evaluation_by_log_prob_score\
                                                , get_only_embeddings, get_embedding_with_probs
import pickle
import math
from scipy import stats
from bias_utils.debiasing_effects_util import get_effect_size, return_log_prob_score, return_seat, load_career_prob
import transformers
import os
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def cossim(x, y):
    return np.dot(x, y) / math.sqrt(np.dot(x, x) * np.dot(y, y))

def load_tmps(temp_file):
    patterns_with_pos = []
    if type(temp_file) is str: 
        print("load tmps from " + temp_file)
        with open(temp_file, 'r', encoding='utf-8') as f:
            for line in f:
                pos = int(line.strip()[0])
                sent = line.strip()[1:]
                patterns_with_pos.append((sent, pos))
    else:
        for line in temp_file:
            pos = int(line.strip()[0])
            sent = line.strip()[1:]
            patterns_with_pos.append((sent, pos))
    return patterns_with_pos

def make_english_row(prof, word, pattern, pos, gender, prof_gender):
    mask = '[MASK]'
    row = []

    # for words such as 'this man' only get 'man'
    if len(word.split()) == 2:
        det = word.split()[0] + " " 
    else:
        det = ""

    if prof[0] in ['e', 'o']:
        if ' a {attr}' in pattern:
            pattern = pattern.replace(" a ", " an ")

    # sentence
    # print(pattern)
    sentence = pattern.format(targ = word, attr = prof)
    row.append(sentence)

    # sentence: masked target
    sent_TM = pattern.format(targ = det + mask, attr = prof)
    row.append(sent_TM)

    # sentence: masked_attribute
    prof_len = len(prof.split())
    sent_AM = pattern.format(targ = word, attr = ' '.join([mask] * prof_len ))
    row.append(sent_AM)

    # sentence: masked target and attribute
    Sent_TAM = pattern.format(targ = det + mask, attr = ' '.join([mask] * prof_len ))
    row.append(Sent_TAM)

    # template
    row.append(pattern.format(targ = '<person subject>', attr = '<profession>'))

    # person:
    if len(word.split()) == 2:
        row.append(word.split()[1])
    else:
        row.append(word)

    # gender
    row.append(gender)

    # profession
    row.append(prof)

    # profession's (statistical) gender
    row.append(prof_gender)

    # pos of predicted target in sent_TM
    row.append(pos)

    # pos of predicted target in Sent_TAM
    need_add = pattern.find('{attr}') < pattern.find('{targ}')
    if need_add:
        Sent_TAM_pos = pos + prof_len
    else:
        Sent_TAM_pos = pos
    row.append(Sent_TAM_pos)

    return row

def make_prof_df(prof_list, sentence_patterns, male_words, female_words, prof_gender):
    data = []

    for sp in sentence_patterns:
        for w in male_words:
            for prof in prof_list:
                gender = 'male'
                row = make_english_row(prof, w, sp[0], sp[1], gender, prof_gender)
                data.append(row)
        for w in female_words:
            for prof in prof_list:
                gender = 'female'
                row = make_english_row(prof, w, sp[0], sp[1], gender, prof_gender)
                data.append(row)

    data = pd.DataFrame(data, columns=['Sentence', 'Sent_TM', 'Sent_AM', 'Sent_TAM',
                                           'Template', 'Person', 'Gender', 'Profession', 'Prof_Gender', 'sent_TM_MASK_pos', 'Sent_TAM_MASK_pos'])

    return data

def convert_tmps_to_sents(temp_file, eval_type, simple = False):

    logging.info("The debiasing effects evaluation type is {}".format(eval_type))

    if simple:
        tmp_type = "simple"
    else:
        tmp_type = "diverse"

    if eval_type == 'test':
        male_words = ['he', 'this man', 'this boy', 'my brother', 'my son', 'my husband', 'my boyfriend', 'my father']
        female_words = ['she', 'this woman', 'this girl', 'my sister', 'my daughter', 'my wife', 'my girlfriend', 'my mother']
        
        with open('data/test_professions_english.json', 'r', encoding='utf-8') as f:
            professions = json.load(f)

    elif eval_type == 'val':
        male_words = ['my uncle', 'my dad']
        female_words = ['my aunt', 'my mom']   

        with open('data/val_professions_english.json', 'r', encoding='utf-8') as f:
            professions = json.load(f)

    else:
        assert eval_type == 'whole'
        male_words = ['he', 'this man', 'this boy', 'my brother', 'my son', 'my husband', 'my boyfriend', 'my father', 'my uncle',
                  'my dad']
        female_words = ['she', 'this woman', 'this girl', 'my sister', 'my daughter', 'my wife', 'my girlfriend', 'my mother',
                        'my aunt', 'my mom']    

        with open('data/professions_english.json', 'r', encoding='utf-8') as f:
            professions = json.load(f)

    patterns = load_tmps(temp_file)
    temp_size = len(patterns)

    corpus = pd.DataFrame()
    for g in ['female', 'male']:
        df = make_prof_df(professions[g], patterns, male_words, female_words, g)
        corpus = corpus.append(df, ignore_index=True)

    # print('The corpus creation was successful!')
    # print('The corpus has a length of {} sentences and {} columns'.format(len(corpus), len(corpus.columns)))

    # corpus.to_csv('../../data/{}_{}_sents_for_evaluating.tsv'.format(tmp_type, temp_size), sep='\t')
    return corpus, tmp_type, temp_size

def dropspace(u, V):
    norm_sqrd = np.sum(V*V, axis=-1)
    vecs = np.divide(V@u, norm_sqrd)[:, None] * V
    subspace = np.sum(vecs, axis=0)
    return u - subspace

def get_embeddings_and_probs(corpus, tmps_type, temp_size, model_type = None, subspace_path = None, tokenizer = None, device = None):
    
    gender_dir = None
    if subspace_path != None:
        if type(subspace_path) is str:
            dir_path = "{}".format(subspace_path)
            print("* load gender dir from ", dir_path)
            with open(dir_path, 'rb') as f:
                gender_dir = pickle.load(f)
        else:
            gender_dir = subspace_path

    AM_list = list(set([i for i in corpus.Sent_AM]))
    
    TM_with_index = list(set(list(zip(corpus.Sent_TM, corpus.sent_TM_MASK_pos))))
    TAM_with_index = list(set(list(zip(corpus.Sent_TAM, corpus.Sent_TAM_MASK_pos))))

    TM_list = [t[0] for t in TM_with_index]
    TM_index_list = [t[1] for t in TM_with_index]

    TAM_list = [t[0] for t in TAM_with_index]
    TAM_index_list = [t[1] for t in TAM_with_index]

    genders_list = list(set([i for i in corpus.Person]))

    all_sents_list = AM_list + TM_list + TAM_list
    print("--- {} sentens to be processes ---".format(len(all_sents_list)))

    sent2emb = {}
    for sent in all_sents_list:
        sent2emb[sent] = {} 

    need_reload = False
    if device is None:
        need_reload = True
        if torch.cuda.is_available():
            device = torch.device('cuda')
            # print('We will use the GPU:', torch.cuda.get_device_name(0))
        else:
            # print('No GPU available, using the CPU instead.')
            device = torch.device('cpu')
    
    if type(model_type) is str:
        tokenizer = BertTokenizer.from_pretrained(model_type)
        logging.info("****** The model is {}".format(model_type))
        model = BertForMaskedLM.from_pretrained(model_type)
    else:
        logging.info("****** The model is already input")
        model = model_type

    max_len_eval = 128
    # print('max_len evaluation: {}'.format(max_len_eval))

    logging.info("******1 The type of used model is {}".format(type(model)))
    tmp_list = TM_list + TAM_list
    all_embeddings, all_probs = get_embedding_with_probs(tmp_list, TM_index_list + TAM_index_list,\
                                        genders_list, tokenizer, model, device, max_len_eval, model_type, gender_dir)
    if gender_dir is not None:
        print("debiasing by sent_debias1...")
        for i, emb in enumerate(all_embeddings):
            emb /= np.linalg.norm(emb)
            emb = dropspace(emb, gender_dir)
            emb /= np.linalg.norm(emb)
            all_embeddings[i] = emb

    for i, _ in enumerate(tmp_list):
        sent2emb[tmp_list[i]]['embeddings'] = all_embeddings[i]
        for j, g in enumerate(genders_list):
            sent2emb[tmp_list[i]][g] = all_probs[i][j]

    all_embeddings = get_only_embeddings(AM_list, tokenizer, model, device, max_len_eval)
    if gender_dir is not None:
        print("debiasing by sent_debias2...")
        for i, emb in enumerate(all_embeddings):
            emb /= np.linalg.norm(emb)
            emb = dropspace(emb, gender_dir)
            emb /= np.linalg.norm(emb)
            all_embeddings[i] = emb
    for i, _ in enumerate(AM_list):
        sent2emb[AM_list[i]]['embeddings'] = all_embeddings[i]

    # with open('./data/res/{}_{}_{}_sent2emb.pkl'.format(model_type, tmps_type, temp_size), 'wb') as f:
    #     pickle.dump(sent2emb, f)
    return sent2emb, model_type

def add_assos(corpus, tmps_type, temp_size, model_type, sent2emb = None):

    # if sent2emb is None:
    #     print("loading sent2emb from saved file ...")
    #     with open('./data/res/{}_{}_{}_sent2emb.pkl'.format(model_type, tmps_type, temp_size), 'wb') as f:
    #         pickle.dump(sent2emb, f)

    seats = []
    log_prob_socres = []
    for index, row in corpus.iterrows():
        TM = row.Sent_TM
        AM = row.Sent_AM
        TAM = row.Sent_TAM
        g = row.Person

        TM_emb = sent2emb[TM]['embeddings']
        AM_emb = sent2emb[AM]['embeddings']
        seat = cossim(TM_emb, AM_emb)
        seats.append(seat)

        target = sent2emb[TM][g]
        prior = sent2emb[TAM][g]
        log_prob_score = np.log(target / prior)
        log_prob_socres.append(log_prob_score)

    corpus = corpus.assign(seat = seats)
    corpus = corpus.assign(log_prob_score = log_prob_socres)
    # corpus.to_csv('./data/res/{}_{}_{}_res.tsv'.format(model_type, tmps_type, temp_size), sep='\t')
    return corpus

if __name__ == '__main__':
    transformers.logging.ERROR

    # model_types = [None, 'sent_debias', 'CDS/CDS_13436_flipped_gap1', 'ER/ER_1_13436_flipped_gap0', 'bleached_ER/bleached_ER_1_13436_flipped_gap3']
 
    p_dir = '../debiased_models/'
    dirs = [d for d in os.listdir(p_dir)]
    dirs.insert(0, None) 
    model_types = dirs

    model2effetcs = {}
    tmp_path = '../data/check_collected_sents_1387.txt'
    # tmp_path = '../data/test_tmps.txt'

    for model_type in model_types:
        print("* Evaluating ", model_type)
        print("* using tmp from", tmp_path)
        model2effetcs[model_type] = {} 
        corpus, tmps_type, temp_size = convert_tmps_to_sents(tmp_path,\
                                                         simple = False)
        sent2emb, model_type = get_embeddings_and_probs(corpus, tmps_type, temp_size, model_type = model_type)
        corpus = add_assos(corpus, tmps_type, temp_size, model_type, sent2emb = sent2emb)

        # formating outputs
        seat_effect_size, log_effect_size = get_effect_size(corpus)
        model2effetcs[model_type]['seat_effect_size'] = seat_effect_size
        model2effetcs[model_type]['log_effect_size'] = log_effect_size

        log_prob_score = return_log_prob_score(corpus)
        seat = return_seat(corpus)
        profs2prob = load_career_prob()

        # print("For log probability score")
        real_probs = []
        computed_assos = []
        for key in ['male', 'female', 'balanced']:
            for pair in log_prob_score[key]:
                prof = pair[0]
                real_probs.append(profs2prob[prof])
                computed_assos.append(pair[1])
        model2effetcs[model_type]['log_corr'] = stats.spearmanr(real_probs, computed_assos)

        # print("For seat")
        real_probs = []
        computed_assos = []
        for key in ['male', 'female', 'balanced']:
            for pair in seat[key]:
                prof = pair[0]
                real_probs.append(profs2prob[prof])
                computed_assos.append(pair[1])
        model2effetcs[model_type]['seat_corr'] = stats.spearmanr(real_probs, computed_assos)
    
    print("* res: ", model2effetcs)
    with open('res/model2effetcs.pkl', 'wb') as f:
        pickle.dump(model2effetcs, f)

