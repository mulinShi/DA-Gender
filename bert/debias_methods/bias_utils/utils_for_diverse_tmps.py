import json
import numpy as np
import torch
from keras.preprocessing.sequence import pad_sequences
from scipy import stats
from torch.nn.functional import softmax
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from transformers import PreTrainedTokenizer
import pandas as pd
import pickle
from bias_utils.utils import input_pipeline
import math
from tqdm import tqdm, trange

def cossim(x, y):
    return np.dot(x, y) / math.sqrt(np.dot(x, x) * np.dot(y, y))

def prob_with_prior(pred_TM, pred_TAM, eval_tokens_TM, eval_tokens_TAM,
                     targs, sent_TM_MASK_pos, sent_TAM_MASK_pos, tokenizer):
    pred_TM = pred_TM.cpu()
    pred_TAM = pred_TAM.cpu()
    targs = targs.cpu()
    eval_tokens_TM = eval_tokens_TM.cpu()
    eval_tokens_TAM = eval_tokens_TAM.cpu()
    sent_TM_MASK_pos = sent_TM_MASK_pos.cpu()
    sent_TAM_MASK_pos = sent_TAM_MASK_pos.cpu()

    probs = []

    for sent_id, _ in enumerate(targs):
        TM_mask_id = sent_TM_MASK_pos[sent_id].item()
        TM_mask_index = (eval_tokens_TM[sent_id] == tokenizer.mask_token_id).nonzero().flatten().tolist()[TM_mask_id]

        TAM_mask_id = sent_TAM_MASK_pos[sent_id].item()
        TAM_mask_index = (eval_tokens_TAM[sent_id] == tokenizer.mask_token_id).nonzero().flatten().tolist()[TAM_mask_id]

        targ = targs[sent_id].item()

        target_prob = pred_TM[sent_id][TM_mask_index][targ].item()
        prior = pred_TAM[sent_id][TAM_mask_index][targ].item()

        probs.append(np.log(target_prob / prior))

    return probs

def model_evaluation_by_log_prob_score(eval_df, tokenizer, model, device):
    max_len = max([len(sent.split()) for sent in eval_df.Sent_TAM])
    pos = math.ceil(math.log2(max_len))
    max_len_eval = int(math.pow(2, pos))

    print('max_len evaluation: {}'.format(max_len_eval))

    eval_tokens_TM, eval_attentions_TM = input_pipeline(eval_df.Sent_TM,
                                                        tokenizer,
                                                        max_len_eval)
    eval_tokens_TAM, eval_attentions_TAM = input_pipeline(eval_df.Sent_TAM,
                                                          tokenizer,
                                                          max_len_eval)
    targs = tokenizer.convert_tokens_to_ids(eval_df.Person)
    targs = torch.tensor(targs).to(torch.int64).unsqueeze(-1)

    sent_TM_MASK_pos = torch.tensor(eval_df.sent_TM_MASK_pos).to(torch.int64).unsqueeze(-1)

    Sent_TAM_MASK_pos = torch.tensor(eval_df.Sent_TAM_MASK_pos).to(torch.int64).unsqueeze(-1)

    # check that lengths match before going further
    assert eval_tokens_TM.shape == eval_attentions_TM.shape
    assert eval_tokens_TAM.shape == eval_attentions_TAM.shape

    # make a Evaluation Dataloader
    eval_batch = 4
    eval_data = TensorDataset(eval_tokens_TM, eval_attentions_TM,
                              eval_tokens_TAM, eval_attentions_TAM,
                              targs, sent_TM_MASK_pos, Sent_TAM_MASK_pos)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, batch_size=eval_batch,
                                 sampler=eval_sampler)

    # put everything to GPU (if it is available)
    # eval_tokens_TM = eval_tokens_TM.to(device)
    # eval_attentions_TM = eval_attentions_TM.to(device)
    # eval_tokens_TAM = eval_tokens_TAM.to(device)
    # eval_attentions_TAM = eval_attentions_TAM.to(device)
    model.to(device)

    # put model in evaluation mode & start predicting
    model.eval()
    associations_all = []
    for step, batch in enumerate(eval_dataloader):
        # print('-- step {} --'.format(step))
        b_input_TM = batch[0].to(device)
        b_att_TM = batch[1].to(device)
        b_input_TAM = batch[2].to(device)
        b_att_TAM = batch[3].to(device)

        with torch.no_grad():
            outputs_TM = model(b_input_TM,
                               attention_mask=b_att_TM)
            outputs_TAM = model(b_input_TAM,
                                attention_mask=b_att_TAM)
            predictions_TM = softmax(outputs_TM[0], dim=2)
            predictions_TAM = softmax(outputs_TAM[0], dim=2)

        assert predictions_TM.shape == predictions_TAM.shape

        # calculate associations
        associations = prob_with_prior(predictions_TM,
                                       predictions_TAM,
                                       b_input_TM,
                                       b_input_TAM, 
                                       batch[4],
                                       batch[5],
                                       batch[6],
                                       tokenizer)

        associations_all += associations

    return associations_all

def model_evaluation_by_seat(eval_df, tokenizer, model, device):
    max_len = max([len(sent.split()) for sent in eval_df.Sent_TAM])
    pos = math.ceil(math.log2(max_len))
    max_len_eval = int(math.pow(2, pos))

    print('max_len evaluation: {}'.format(max_len_eval))

    eval_tokens_TM, eval_attentions_TM = input_pipeline(eval_df.Sent_TM,
                                                        tokenizer,
                                                        max_len_eval)
    eval_tokens_AM, eval_attentions_AM = input_pipeline(eval_df.Sent_AM,
                                                          tokenizer,
                                                          max_len_eval)

    # check that lengths match before going further
    assert eval_tokens_TM.shape == eval_attentions_TM.shape
    assert eval_tokens_AM.shape == eval_tokens_AM.shape

    # make a Evaluation Dataloader
    eval_batch = 20
    eval_data = TensorDataset(eval_tokens_TM, eval_attentions_TM,
                              eval_tokens_AM, eval_attentions_AM)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, batch_size=eval_batch,
                                 sampler=eval_sampler)

    model.to(device)

    # put model in evaluation mode & start predicting
    model.eval()
    associations_all = []
    for step, batch in enumerate(eval_dataloader):
        # print('-- step {} --'.format(step))
        b_input_TM = batch[0].to(device)
        b_att_TM = batch[1].to(device)
        b_input_AM = batch[2].to(device)
        b_att_AM = batch[3].to(device)

        with torch.no_grad():
            TM_embs = model(b_input_TM, attention_mask=b_att_TM,
                output_hidden_states=True).hidden_states[-1]
            TM_embs = torch.mean(TM_embs, dim=1).cpu().detach().numpy()

            AM_embs = model(b_input_AM, attention_mask=b_att_AM,
                output_hidden_states=True).hidden_states[-1]
            AM_embs = torch.mean(AM_embs, dim=1).cpu().detach().numpy()

        # calculate associations
        associations = [cossim(TM_embs[i], AM_embs[i]) for i, _ in enumerate(TM_embs)]
        associations_all += associations

    return associations_all

def get_embs_for_autoregressive_models(sents, tokenizer, model, device):

    model.to(device)
    model.eval()

    all_embeddings = []
    for i, sent in enumerate(tqdm(sents)):
        tokens = tokenizer.encode(sents[i])
        tokens_tensor = torch.tensor(
                    tokens).to(device).unsqueeze(0)
        with torch.no_grad():
            embs = model(tokens_tensor, output_hidden_states=True).hidden_states[-1][:, -1, :].cpu().detach().numpy()
        assert list(embs.shape) == [1, 768]
        all_embeddings.append(embs)
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    return all_embeddings

def get_only_embeddings(sents, tokenizer, model, device, max_len):

    print("--- get embeddings from AM---")

    eval_tokens, eval_attentions = input_pipeline(sents, tokenizer, max_len)
                                                        
    # check that lengths match before going further
    assert eval_tokens.shape == eval_attentions.shape

    # make a Evaluation Dataloader
    eval_batch = 1
    eval_data = TensorDataset(eval_tokens, eval_attentions)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, batch_size=eval_batch,
                                 sampler=eval_sampler)

    model.to(device)

    # put model in evaluation mode & start predicting
    model.eval()
    all_embeddings = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        # print('-- step {} --'.format(step))
        b_input = batch[0].to(device)
        b_att = batch[1].to(device)

        with torch.no_grad():
            embs = model(b_input, attention_mask=b_att,
                    output_hidden_states=True).hidden_states[-1]
            embs = torch.mean(embs, dim=1).cpu().detach().numpy()
            # if model_type == 'bert':
            #     embs = model(b_input, attention_mask=b_att,
            #         output_hidden_states=True).hidden_states[-1]
            #     embs = torch.mean(embs, dim=1).cpu().detach().numpy()
            # else:
            #     assert model_type == 'bart' or model_type == 'gpt2'
            #     embs = model(b_input, attention_mask=b_att,
            #         output_hidden_states=True).hidden_states[-1][:, -1, :]
            #     assert list(embs.shape) == [1, 768]
            all_embeddings.append(embs)

    all_embeddings = np.concatenate(all_embeddings, axis=0)

    return all_embeddings

def dropspace(u, V):
    norm_sqrd = np.sum(V*V, axis=-1)
    vecs = np.divide(V@u, norm_sqrd)[:, None] * V
    subspace = np.sum(vecs, axis=0)
    return u - subspace

def get_embedding_with_probs(sents, indexs, genders_list, tokenizer, model, device, max_len_eval, model_type, gender_dir):

    print("--- get embeddings from TM and TAM---")

    eval_tokens, eval_attentions = input_pipeline(sents,
                                                        tokenizer,
                                                        max_len_eval)

    targets = tokenizer.convert_tokens_to_ids(genders_list)
    targets = np.array(targets)

    indexs = torch.tensor(indexs).to(torch.int64).unsqueeze(-1)

    # check that lengths match before going further
    assert eval_tokens.shape == eval_attentions.shape

    # make a Evaluation Dataloader
    eval_batch = 4
    eval_data = TensorDataset(eval_tokens, eval_attentions,
                              indexs)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, batch_size=eval_batch,
                                 sampler=eval_sampler)

    model.to(device)

    # put model in evaluation mode & start predicting
    model.eval()
    all_embeddings = []
    all_probs = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        # print('-- step {} --'.format(step))
        b_input = batch[0].to(device)
        b_att = batch[1].to(device)

        with torch.no_grad():
            outputs = model(b_input, attention_mask=b_att,
                output_hidden_states=True)

            embs = outputs.hidden_states[-1]
            embs = torch.mean(embs, dim=1).cpu().detach().numpy()
            all_embeddings.append(embs)
            outputs = outputs[0]

            if gender_dir is not None:
                # print("* computing debiased log probability score")
                outputs_before_cls = model.bert(b_input, attention_mask=b_att).last_hidden_state
                outputs_before_cls = outputs_before_cls.cpu().detach().numpy()
                for b in range(outputs_before_cls.shape[0]):
                    for i in range(outputs_before_cls.shape[1]):
                        emb = outputs_before_cls[b, i, :]
                        emb /= np.linalg.norm(emb)
                        emb = dropspace(emb, gender_dir)
                        emb /= np.linalg.norm(emb)
                        outputs_before_cls[b, i, :] = emb
                outputs_before_cls = torch.tensor(outputs_before_cls).to(device)
                outputs = model.cls(outputs_before_cls)

            predictions = softmax(outputs, dim=2)

        # calculate associations
        probs = get_probs(predictions,
                                       b_input,
                                       batch[2],
                                       targets,
                                       tokenizer)
        all_probs += probs
    all_embeddings = np.concatenate(all_embeddings, axis=0)

    return all_embeddings, all_probs

def get_probs(predictions, sents, indexs, targts, tokenizer):
    predictions = predictions.cpu().detach().numpy()
    sents = sents.cpu()
    indexs = indexs.cpu()

    probs = []

    for sent_id, _ in enumerate(predictions):

        masked_id = indexs[sent_id].item()
        mask_index = (sents[sent_id] == tokenizer.mask_token_id).nonzero().flatten().tolist()[masked_id]

        probs.append(predictions[sent_id][mask_index][targts])

    return probs
