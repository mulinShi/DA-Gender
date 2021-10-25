import numpy as np
import json
import pandas as pd
from scipy import stats

def load_json(file_name):
    with open(file_name, 'r') as f:
        j = json.load(f)
    return j

def read_tsv(file_name):
    return pd.read_csv(file_name, sep='\t', index_col=0)

def return_log_prob_score(res):
    dic = {}
    for prof_gender in ['male', 'female', 'balanced']:
        cur_df = res.groupby('Profession')
        assos = []
        for prof in cur_df.groups:
            male_ass = cur_df.get_group(prof).loc[res.Gender == 'male'].log_prob_score.mean()
            female_ass = cur_df.get_group(prof).loc[res.Gender == 'female'].log_prob_score.mean()
            assos.append((prof, male_ass - female_ass))
        dic[prof_gender] = assos
    return dic

def return_seat(res):
    dic = {}
    for prof_gender in ['male', 'female', 'balanced']:
        cur_df = res.groupby('Profession')
        assos = []
        for prof in cur_df.groups:
            male_ass = cur_df.get_group(prof).loc[res.Gender == 'male'].seat.mean()
            female_ass = cur_df.get_group(prof).loc[res.Gender == 'female'].seat.mean()
            assos.append((prof, male_ass - female_ass))
        dic[prof_gender] = assos
    return dic

def print_prof_gender_res(res):
    for prof_gender in ['male', 'female', 'balanced']:
        male = res.loc[res.Prof_Gender == prof_gender].loc[res.Gender == 'male'].log_prob_score.mean()
        female = res.loc[res.Prof_Gender == prof_gender].loc[res.Gender == 'female'].log_prob_score.mean()
        
        male1 = res.loc[res.Prof_Gender == prof_gender].loc[res.Gender == 'male'].seat.mean()
        female1 = res.loc[res.Prof_Gender == prof_gender].loc[res.Gender == 'female'].seat.mean()
        
        print("{} jobs + {} log probabilty score mean is : {} seat mean is : {}".format(prof_gender, 'male', male, male1))
        print("{} jobs + {} log probabilty score mean is : {} seat mean is : {}".format(prof_gender, 'female', female, female1))
        print("log probabilty score dif is {} seat dif is {}".format(male - female, male1 - female1))

def load_career_prob(filname = '../data/Professions US+DE.tsv'):
    profs = read_tsv(filname)[['simplified_professions', 'percent_us']]
    profs = list(zip(profs['simplified_professions'].tolist(), profs['percent_us'].tolist()))
    profs2prob = {}
    for i, pair in enumerate(profs):
        profs2prob[pair[0]] = 100 - pair[1]
    return profs2prob

def get_effect_size(corpus):
    males = list(corpus.loc[corpus.Gender == 'male'].groupby('Person').groups)
    females = list(corpus.loc[corpus.Gender == 'female'].groupby('Person').groups)
    
    male_seat = []
    female_seat = []
    male_log = []
    female_log = []
    
    for male in males:
        sents = corpus.loc[corpus.Person == male]
        male_seat.append(sents.loc[sents.Prof_Gender == 'male'].seat.mean() \
                         - sents.loc[sents.Prof_Gender == 'female'].seat.mean())
        male_log.append(sents.loc[sents.Prof_Gender == 'male'].log_prob_score.mean() \
                         - sents.loc[sents.Prof_Gender == 'female'].log_prob_score.mean())
        
    for female in females:
        sents = corpus.loc[corpus.Person == female]
        female_seat.append(sents.loc[sents.Prof_Gender == 'male'].seat.mean() \
                 - sents.loc[sents.Prof_Gender == 'female'].seat.mean())
        female_log.append(sents.loc[sents.Prof_Gender == 'male'].log_prob_score.mean() \
                         - sents.loc[sents.Prof_Gender == 'female'].log_prob_score.mean())
    
    male_seat = np.array(male_seat)
    female_seat = np.array(female_seat)
    male_log = np.array(male_log)
    female_log = np.array(female_log)
    
    diff = (male_seat.mean() - female_seat.mean())
    std_ = np.concatenate([male_seat, female_seat], axis=0).std() + 1e-8
#     print(np.concatenate([male_seat, female_seat], axis=0).shape)
    seat_effect_size = diff / std_
#     print(diff, ' ', std_)
    seat_p = exact_mc_perm_test(male_seat, female_seat)
    # print("for seat effect size is {} p value is {}".format(effect_size, p))
    
    diff = (male_log.mean() - female_log.mean())
    std_ = np.concatenate([male_log, female_log], axis=0).std() + 1e-8
    log_effect_size = diff / std_
    # print(diff, ' ', std_)
    log_p = exact_mc_perm_test(male_log, female_log)
    # print("for log probability score effect size is {} p value is {}".format(effect_size, p))
    return seat_effect_size, seat_p, log_effect_size, log_p


def exact_mc_perm_test(xs, ys, nmc=100000):
    n, k = len(xs), 0
    diff = np.abs(np.mean(xs) - np.mean(ys))
    zs = np.concatenate([xs, ys])
    for j in range(nmc):
#         print(j)
        np.random.shuffle(zs)
        k += diff < np.abs(np.mean(zs[:n]) - np.mean(zs[n:]))
    return k / nmc