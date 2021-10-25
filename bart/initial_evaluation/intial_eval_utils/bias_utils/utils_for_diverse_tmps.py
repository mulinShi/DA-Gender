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
from bias_utils.utils import new_input_pipeline
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

def get_only_embeddings(sents, tokenizer, model, device, max_len = 128):

    print("--- get embeddings from AM---")

    eval_tokens, eval_attentions = new_input_pipeline(sents, tokenizer, max_len)
                                                        
    # check that lengths match before going further
    # assert eval_tokens.shape == eval_attentions.shape

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
                    output_hidden_states=True).decoder_hidden_states[-1]
            # embs = embs.cpu().detach().numpy()[:, -1, :]
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

    eval_tokens, eval_attentions = new_input_pipeline(sents, tokenizer, max_len_eval)

    targets = tokenizer.convert_tokens_to_ids(genders_list)
    targets = np.array(targets)

    indexs = torch.tensor(indexs).to(torch.int64).unsqueeze(-1)

    # check that lengths match before going further
    # assert eval_tokens.shape == eval_attentions.shape

    # make a Evaluation Dataloader
    eval_batch = 1
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

            embs = outputs.decoder_hidden_states[-1]

            embs = torch.mean(embs, dim=1).cpu().detach().numpy()
            # embs = embs.cpu().detach().numpy()[:, -1, :]

            all_embeddings.append(embs)
            outputs = outputs[0]

            if gender_dir is not None:
                # print("* computing debiased log probability score")
                outputs_before_cls = model.model(b_input, attention_mask=b_att).last_hidden_state
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
