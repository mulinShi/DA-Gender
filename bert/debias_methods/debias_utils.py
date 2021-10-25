import datetime
import math
from typing import Tuple
import json
import numpy as np
import torch
from keras.preprocessing.sequence import pad_sequences
from scipy import stats
from torch.nn.functional import softmax
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from transformers import PreTrainedTokenizer
from transformers import BertTokenizer, BertForMaskedLM
import pandas as pd

def save_model(tokenizer, model, loss_log = None, save_directory = '../debiased_models/', debias_type = 'CDS', repeat_time = None):
    file = save_directory + debias_type + str(repeat_time)
    tokenizer.save_pretrained(file)
    model.save_pretrained(file)
    print("saving model to {}".format(file))

    if loss_log is not None:
        print("saving loss log to {}".format(file))
        with open (file + '/loss_log', 'w') as l:
            for loss in loss_log:
                l.write(str(loss) + '\n')
    return   
  
def save_model_for_trade_off(tokenizer, model, debias_type = 'CDS', repeat_time = None):
    file = debias_type + str(repeat_time)
    tokenizer.save_pretrained(file)
    model.save_pretrained(file)
    print("saving model to {}".format(file))
    return   

def load_model(save_directory = '../', debias_type = 'CDS'):
    file = save_directory + "{}_debiased".format(debias_type)
    print("loading model from {}".format(file))
    tokenizer = BertTokenizer.from_pretrained(file)
    model = BertForMaskedLM.from_pretrained(file,
                                            output_attentions=False,
                                            output_hidden_states=False)
    return tokenizer, model


def format_time(elapsed):
    """ Takes a time in seconds and returns a string hh:mm:ss"""
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def tokenize_to_id(sentences, tokenizer):
    """Tokenize all of the sentences and map the tokens to their word IDs."""
    input_ids = []

    # For every sentence...
    for sent in sentences:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        encoded_sent = tokenizer.encode(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'

            # This function also supports truncation and conversion
            # to pytorch tensors, but we need to do padding, so we
            # can't use these features :( .
            # max_length = 128,          # Truncate all sentences.
            # return_tensors = 'pt',     # Return pytorch tensors.
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_sent)
    return input_ids

def attention_mask_creator(input_ids):
    """Provide the attention mask list of lists: 0 only for [PAD] tokens (index 0)
    Returns torch tensor"""
    attention_masks = []
    for sent in input_ids:
        segments_ids = [int(t > 0) for t in sent]
        attention_masks.append(segments_ids)
    return torch.tensor(attention_masks)

def CDS_input_pipeline(sequence, tokenizer, MAX_LEN):
    """function to tokenize, pad and create attention masks"""
    input_ids = tokenize_to_id(sequence, tokenizer)
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN,
                              dtype="long",
                              value=tokenizer.pad_token_id,
                              truncating="post", padding="post")
    input_ids = torch.tensor(input_ids).to(torch.int64)

    attention_masks = attention_mask_creator(input_ids).to(torch.int64)

    return input_ids, attention_masks

def RE_input_pipeline(sequence, tokenizer, MAX_LEN, substitutor):
    print("Preparing inputs for ER...")
    """function to tokenize, pad and create attention masks"""
    flipped_seq = []
    for i, sent in enumerate(sequence):
        flipped_sent, cnt = substitutor.invert_document(sequence[i])
        flipped_seq.append(flipped_sent)

        # assert cnt > 0, "Error2: {}".format(sent)

        # tokenized_sent = tokenizer.tokenize(sequence[i])
        # tokenized_flipped_sent = tokenizer.tokenize(flipped_sent)

        # assert len(tokenized_sent) == len(tokenized_flipped_sent),\
        #  "Error1: tokenized seq size is different for: {}".format(sent)

        # diff_cnt = sum([1 for i in range(len(tokenized_sent)) if tokenized_sent[i] != tokenized_flipped_sent[i]])
        # assert diff_cnt == cnt, "Error2: tokenized seq have incorrect number of differet tokens for: {}".format(sent)

    input_ids = tokenize_to_id(sequence, tokenizer)
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN,
                              dtype="long",
                              value=tokenizer.pad_token_id,
                              truncating="post", padding="post")
    input_ids = torch.tensor(input_ids).to(torch.int64)
    attention_masks = attention_mask_creator(input_ids).to(torch.int64)

    flipped_input_ids = tokenize_to_id(flipped_seq, tokenizer)
    flipped_input_ids = pad_sequences(flipped_input_ids, maxlen=MAX_LEN,
                              dtype="long",
                              value=tokenizer.pad_token_id,
                              truncating="post", padding="post")
    flipped_input_ids = torch.tensor(flipped_input_ids).to(torch.int64)
    flipped_attention_masks = attention_mask_creator(flipped_input_ids).to(torch.int64)
    print("Finished preparing inputs...")
    return input_ids, attention_masks, flipped_input_ids, flipped_attention_masks


def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, mlm_probability=0.15) -> Tuple[
    torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the "
            "--mlm flag if you want to use this tokenizer. "
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults
    # to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]

    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

def mask_tokens_for_RE(inputs1: torch.Tensor, inputs2: torch.Tensor, tokenizer: PreTrainedTokenizer, mlm_probability=0.15) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the "
            "--mlm flag if you want to use this tokenizer. "
        )

    assert inputs1.shape == inputs2.shape

    labels1 = inputs1.clone()
    labels2 = inputs2.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults
    # to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels1.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels1.tolist()
    ]

    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels1.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()

    # We only compute loss on masked tokens
    labels1[~masked_indices] = -100  
    labels2[~masked_indices] = -100

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels1.shape, 0.8)).bool() & masked_indices
    inputs1[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    inputs2[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels1.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels1.shape, dtype=torch.long)
    inputs1[indices_random] = random_words[indices_random]
    inputs2[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs1, labels1, inputs2, labels2, masked_indices
