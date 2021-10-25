import sys 
import json
import pandas as pd
import os
import numpy as np
import argparse
sys.path.append("intial_eval_utils")

from intial_eval_utils.evaluating_over_debiasing_bart import add_predicts_bart, get_vios_and_dif
from intial_eval_utils.evaluate_by_divers_tmps_bart import convert_tmps_to_sents, get_embeddings_and_probs_bart, add_assos
from intial_eval_utils.bias_utils.debiasing_effects_util import get_effect_size
from intial_eval_utils.bart_ori_seat import get_embeddings, compute_effect_size

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import pickle

eval_type = 'test'

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', required = True)    
    parser.add_argument('--models_dir', help='which model to use', required=False, default = 'debiased_models/')
    parser.add_argument('--debiasing_type', help='', required=False, default = 'CDS')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_arguments()
    models_dir = args.models_dir
    debiasing_type = args.debiasing_type
    output_file = args.output_file
    
    dirs = [d for d in os.listdir(models_dir + debiasing_type + '/')]
    logging.info("Models to be processed are {}".format(dirs))

    res = {}

    res['model_names'] = []

    res['err_nums'] = []
    res['IAs'] = []

    for m in dirs:

        res['model_names'].append(m)

        model = models_dir + debiasing_type + '/' + m

        logging.info("* Starting evaluating over-debiasing for model {}".format(m))

        data_without_preds_path = '../data/bart_2610_sents_for_evaluating_gender_loss.tsv'

        data_without_preds = pd.read_csv(data_without_preds_path, sep='\t', index_col = 0)

        data = add_predicts_bart(data_without_preds, model_type = model)

        violates, difs = get_vios_and_dif(data)

        data.to_csv('{}{}_corpus.tsv'.format(output_file, m), sep='\t')
    
        error_num = len(violates)
        res['err_nums'].append(error_num)

        IA = sum(difs) / len(difs)
        res['IAs'].append(IA)

    accs = [1.0 - err_cnt / 2610 for err_cnt in res['err_nums']]

    res['stat_acc'] = (np.array(accs).mean(), np.array(accs).std())
    res['stat_IA'] = (np.array(res['IAs']).mean(), np.array(res['IAs']).std())

    with open('{}{}_res.json'.format(output_file, debiasing_type), 'w') as f:
        json.dump(res, f, indent = 6)
