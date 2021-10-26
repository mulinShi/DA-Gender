import sys 
import json
import pandas as pd
import argparse
sys.path.append("eval_utils")

from eval_utils.evaluating_over_debiasing_bart import add_predicts_bart, get_vios_and_dif
from eval_utils.seat_v2_and_logprob import convert_tmps_to_sents, get_embeddings_and_probs_bart, add_assos
from eval_utils.bias_utils.debiasing_effects_util import get_effect_size
from eval_utils.seat_v1 import get_embeddings, compute_effect_size

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='which model to use', required=False, default = 'facebook/bart-base')
    parser.add_argument('--eval_type', required=False, default = 'test')
    parser.add_argument('--subspace_path', help='gender subspace of Sentence Debias', required=False, default = None)
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_arguments()

    eval_type = args.eval_type
    model_path = args.model_path
    subspace_path = args.subspace_path

    res = {}
    res['model'] = model_path
    if subspace_path is not None:
        res['model'] += "_SENT_DEBIAS"

    print("* Evaluating via seat-v2 and logprob ...")
    tmp_path = 'data/simple_patterns'

    corpus, tmps_type, temp_size = convert_tmps_to_sents(tmp_path, eval_type,\
                                                     simple = True)
    sent2emb = get_embeddings_and_probs_bart(corpus, tmps_type, temp_size, 
                model_type = model_path, subspace_path = subspace_path)
    corpus = add_assos(corpus, tmps_type, temp_size, sent2emb = sent2emb)

    seat_effect_size, seat_p, log_effect_size, log_p = get_effect_size(corpus)

    res['SEAT-v2'] = seat_effect_size
    res['logprob'] = log_effect_size
    print("For SEAT-v2: effect size is {}, p value is {}".format(seat_effect_size, seat_p))
    print("For log probability score: effect size is {}, p value is {}".format(log_effect_size, log_p))

    print("* Evaluating via seat-v1 ...")
    if eval_type == 'val':
        ori_seat_corpus_path = 'data/val_new_ori_seat_data.json'
    elif eval_type == 'test':
        ori_seat_corpus_path = 'data/test_new_ori_seat_data.json'
    else:
        assert eval_type == 'whole'
        ori_seat_corpus_path = 'data/new_ori_seat_data.json'

    with open(ori_seat_corpus_path, 'r') as f:
        test = json.load(f)

    sent2emb = get_embeddings(test, model_type = model_path, subspace_path = subspace_path)

    seat_ez, p = compute_effect_size(test, sent2emb = sent2emb)

    res['SEAT-v1'] = seat_ez
    print("For SEAT-v1: effect size is {}, p value is {}".format(seat_ez, p))

    print("* Evaluating via DA-score ...")
    data_without_preds_path = 'data/bart_2610_sents_for_evaluating_gender_loss.tsv'

    data_without_preds = pd.read_csv(data_without_preds_path, sep='\t', index_col = 0)

    data = add_predicts_bart(data_without_preds, model_type = model_path, subspace_path = subspace_path)

    violates, difs = get_vios_and_dif(data)

    error_num = len(violates)

    error_rate = len(violates)/len(data)

    res['DA-score'] = 1 - error_rate
    print("For DA-score: ", res['DA-score'])

    with open("bart_bias_eval_res.json", 'a') as f:
        json.dump(res, f, indent = 6)