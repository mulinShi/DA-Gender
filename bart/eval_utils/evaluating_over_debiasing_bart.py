import pandas as pd
import json
import numpy as np
import torch
from transformers import BartTokenizer, BartForConditionalGeneration, BartForCausalLM, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import pickle
import math
import sys
sys.path.append( '../../' )
from tqdm import tqdm, trange
from torch.nn.functional import softmax
import transformers
import os
import argparse
from bias_utils.utils import new_input_pipeline

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

def dropspace(u, V):
    norm_sqrd = np.sum(V*V, axis=-1)
    vecs = np.divide(V@u, norm_sqrd)[:, None] * V
    subspace = np.sum(vecs, axis=0)
    return u - subspace

def get_probs_helper(TM_predictions, TAM_predictions, \
                          TM_input_b, TAM_input_b, male_b, female_b, mask_id_b, tokenizer):
    pref2male_b = []
    pref2female_b = []

    TM_predictions = TM_predictions.cpu().detach().numpy()
    TM_input_b = TM_input_b.cpu()

    TAM_predictions = TAM_predictions.cpu().detach().numpy()
    TAM_input_b = TAM_input_b.cpu()
    mask_id_b = mask_id_b.cpu()

    for b, _ in enumerate(TM_predictions):
        # print(TM_input_b[b])
        TM_mask_pos = (TM_input_b[b] == tokenizer.mask_token_id).nonzero().flatten().tolist()[0]
        TM_prob = TM_predictions[b][TM_mask_pos][male_b[b]]

        masked_id = mask_id_b[b].item()
        TAM_mask_pos = (TAM_input_b[b] == tokenizer.mask_token_id).nonzero().flatten().tolist()[masked_id]
        TAM_prob = TAM_predictions[b][TAM_mask_pos][male_b[b]]

        pref2male_b.append(np.log(TM_prob / TAM_prob))

        TM_mask_pos = (TM_input_b[b] == tokenizer.mask_token_id).nonzero().flatten().tolist()[0]
        TM_prob = TM_predictions[b][TM_mask_pos][female_b[b]]

        masked_id = mask_id_b[b].item()
        TAM_mask_pos = (TAM_input_b[b] == tokenizer.mask_token_id).nonzero().flatten().tolist()[masked_id]
        TAM_prob = TAM_predictions[b][TAM_mask_pos][female_b[b]]
        
        pref2female_b.append(np.log(TM_prob / TAM_prob))

    return pref2male_b, pref2female_b
    
def get_probs(TM_list, TAM_list, male_targets, female_targets,\
                 mask_indexs, tokenizer, model, device, max_len_eval, model_type, gender_dir):

    TM_tokens, TM_attentions = new_input_pipeline(TM_list,
                                                tokenizer,
                                                max_len_eval)

    TAM_tokens, TAM_attentions = new_input_pipeline(TAM_list,
                                                tokenizer,
                                                max_len_eval)

    male_targets = tokenizer.convert_tokens_to_ids(male_targets)
    male_targets = torch.tensor(male_targets).to(torch.int64)

    female_targets = tokenizer.convert_tokens_to_ids(female_targets)
    female_targets = torch.tensor(female_targets).to(torch.int64)

    mask_indexs = torch.tensor(mask_indexs).to(torch.int64)

    # check that lengths match before going further
    assert TAM_tokens.shape == TAM_attentions.shape

    # make a Evaluation Dataloader
    eval_batch = 1
    eval_data = TensorDataset(TM_tokens, TM_attentions,
                              TAM_tokens, TAM_attentions,
                              male_targets, female_targets, 
                              mask_indexs)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, batch_size=eval_batch,
                                 sampler=eval_sampler)

    model.to(device)

    # put model in evaluation mode & start predicting
    model.eval()

    pref2males = []
    pref2females = []
    for step, batch in enumerate(tqdm(eval_dataloader)):

        TM_input_b = batch[0].to(device)
        TM_att_b = batch[1].to(device)

        TAM_input_b = batch[2].to(device)
        TAM_att_b = batch[3].to(device)

        male_b = batch[4].to(device)
        female_b = batch[5].to(device)

        mask_id_b = batch[6].to(device)

        with torch.no_grad():
            # for TM
            outputs = model(input_ids = TM_input_b, attention_mask=TM_att_b,
                output_hidden_states=True)[0]

            if gender_dir is not None:
                # print("* computing debiased log probability score")
                outputs_before_cls = model.model(input_ids = TM_input_b, attention_mask=TM_att_b).last_hidden_state
                outputs_before_cls = outputs_before_cls.cpu().detach().numpy()
                for b in range(outputs_before_cls.shape[0]):
                    for i in range(outputs_before_cls.shape[1]):
                        emb = outputs_before_cls[b, i, :]
                        emb /= np.linalg.norm(emb)
                        emb = dropspace(emb, gender_dir)
                        emb /= np.linalg.norm(emb)
                        outputs_before_cls[b, i, :] = emb
                outputs_before_cls = torch.tensor(outputs_before_cls).to(device)
                outputs = model.lm_head(outputs_before_cls)

            TM_predictions = softmax(outputs, dim=2)

            # for TAM
            outputs = model(input_ids = TAM_input_b, attention_mask=TAM_att_b,
                output_hidden_states=True)[0]

            if gender_dir is not None:
                # print("* computing debiased log probability score")
                outputs_before_cls = model.model(input_ids = TAM_input_b, attention_mask=TAM_att_b).last_hidden_state
                outputs_before_cls = outputs_before_cls.cpu().detach().numpy()
                for b in range(outputs_before_cls.shape[0]):
                    for i in range(outputs_before_cls.shape[1]):
                        emb = outputs_before_cls[b, i, :]
                        emb /= np.linalg.norm(emb)
                        emb = dropspace(emb, gender_dir)
                        emb /= np.linalg.norm(emb)
                        outputs_before_cls[b, i, :] = emb
                outputs_before_cls = torch.tensor(outputs_before_cls).to(device)
                outputs = model.lm_head(outputs_before_cls)

            TAM_predictions = softmax(outputs, dim=2)

        # calculate associations
        pref2male_b, pref2female_b = get_probs_helper(TM_predictions, TAM_predictions, \
                          TM_input_b, TAM_input_b, male_b, female_b, mask_id_b, tokenizer)

        pref2males += pref2male_b
        pref2females += pref2female_b

    return pref2males, pref2females

def add_predicts_bart(corpus, device = None, model_type = None, subspace_path = None, output_name = ""):
    
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

    TM_list = [i for i in corpus.TM_sent]
    TAM_list = [i for i in corpus.TAM_sent]
    male_targets = [i for i in  corpus.Male_target]
    female_targets = [i for i in corpus.Female_target]

    mask_indexs = []
    for i, tmp in enumerate(corpus.Used_tmp):
        target_pos = tmp.find("arget>")
        attr_pos = tmp.find("_attr>")
        if attr_pos == -1:
            attr_pos = tmp.find("<attr's>")
        assert target_pos != -1 and attr_pos != -1
        index = int(target_pos > attr_pos)
        mask_indexs.append(index)
    print("--- {} sentens to be processes ---".format(len(TM_list + TAM_list)))

    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print('We will use the GPU:', torch.cuda.get_device_name(0))
        else:
            print('No GPU available, using the CPU instead.')
            device = torch.device('cpu')

    if type(model_type) is str:
        print("Loading the tokenizer and the model from {}".format(model_type))
        tokenizer = BartTokenizer.from_pretrained(model_type)
        model = BartForConditionalGeneration.from_pretrained(model_type)
    else:
        model = model_type

    max_len_eval = 128
    print('max_len evaluation: {}'.format(max_len_eval))

    pref2males, pref2females = get_probs(TM_list, TAM_list,\
                                       male_targets, female_targets,\
                                       mask_indexs, tokenizer, model, device, max_len_eval, model_type, gender_dir)

    corpus = corpus.assign(pref_to_male = pref2males)
    corpus = corpus.assign(pref_to_female = pref2females)
    
    # corpus.to_csv('sent_debias_large_bart_{}_{}_over_debiasing.tsv'.format(len(corpus), output_name), sep='\t')
    return corpus

def parse_arguments():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--repeat_times', required=True, default = 1)
    parser.add_argument('--model_type', help='which model to use', required=False, default = 'bart')
    parser.add_argument('--debiasing_type', help='', required=False, default = 'None')
    parser.add_argument('--subspace_path', help='', required=False, default = None)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    transformers.logging.ERROR

    args = parse_arguments()
    model_type = args.model_type
    debiasing_type = args.debiasing_type

    data_without_preds_path = 'data/bart_2610_sents_for_evaluating_gender_loss.tsv'

    over_debias = {}

    data_without_preds = pd.read_csv(data_without_preds_path, sep='\t', index_col = 0)

    data = add_predicts_bart(data_without_preds, model_type = model_type)
    violates, difs = get_vios_and_dif(data)

    error_num = len(violates)
    over_debias['error_num'] = error_num

    error_rate = len(violates)/len(data)
    over_debias['error_rate'] = error_rate

    IA = sum(difs) / len(difs)
    over_debias['IA'] = IA

    with open('res/{}_over_debiasing.json'.format(debiasing_type), 'w') as f:
        json.dump(over_debias, f, indent = 6)
