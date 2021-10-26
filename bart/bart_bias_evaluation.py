import sys 
import json
import pandas as pd

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

    model_name = model_path.split('/')[-1]

    print("* Evaluating via seat-v2 and logprob ...")

    tmp_path = 'data/simple_patterns'

    corpus, tmps_type, temp_size = convert_tmps_to_sents(tmp_path, eval_type,\
                                                     simple = True)
    sent2emb = get_embeddings_and_probs_bart(corpus, tmps_type, temp_size, 
                model_type = model_path, subspace_path = subspace_path)
    corpus = add_assos(corpus, tmps_type, temp_size, sent2emb = sent2emb)

    seat_effect_size, seat_p, log_effect_size, log_p = get_effect_size(corpus)
    print("For SEAT: effect size is {}, p value is {}".format(seat_effect_size, seat_p))
    print("For log probability score: effect size is {}, p value is {}".format(log_effect_size, log_p))