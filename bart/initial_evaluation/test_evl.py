import sys 
import json
import pandas as pd

sys.path.append("intial_eval_utils")

from intial_eval_utils.evaluating_over_debiasing_bart import add_predicts_bart, get_vios_and_dif
from intial_eval_utils.evaluate_by_divers_tmps_bart import convert_tmps_to_sents, get_embeddings_and_probs_bart, add_assos
from intial_eval_utils.bias_utils.debiasing_effects_util import get_effect_size
from intial_eval_utils.bart_ori_seat import get_embeddings, compute_effect_size

eval_type = 'test'

tmp_path = '../data/simple_patterns'

corpus, tmps_type, temp_size = convert_tmps_to_sents(tmp_path, eval_type,\
                                                 simple = True)
sent2emb = get_embeddings_and_probs_bart(corpus, tmps_type, temp_size, model_type = 'facebook/bart-base')
corpus = add_assos(corpus, tmps_type, temp_size, sent2emb = sent2emb)

seat_effect_size, seat_p, log_effect_size, log_p = get_effect_size(corpus)
print("For SEAT: effect size is {}, p value is {}".format(seat_effect_size, seat_p))
print("For log probability score: effect size is {}, p value is {}".format(log_effect_size, log_p))