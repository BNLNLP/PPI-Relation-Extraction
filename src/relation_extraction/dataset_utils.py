import os
import sys
import re
import pickle
import json
import pandas as pd
import numpy as np
from datasets import ClassLabel, load_dataset, load_metric, Dataset, DatasetDict, concatenate_datasets
from transformers import DebertaV2Tokenizer

from transformers import BertTokenizerFast, RobertaTokenizerFast

import spacy
from spacy.symbols import ORTH
nlp = spacy.load("en_core_web_sm", disable=["tagger", "ner"])

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def tokenize_and_set_relation_labels(examples, tokenizer, padding, max_seq_length, relation_representation, use_entity_typed_marker, use_context):

    if 'text' in examples:
        if relation_representation.startswith('EM'):
            token_key = 'text_with_typed_entity_marker' if use_entity_typed_marker else 'text_with_entity_marker'
        else:
            token_key = 'text'
        
        tokenized_inputs = tokenizer(
            examples[token_key],
            padding=padding,
            truncation=True,
            max_length=max_seq_length,
        )
    elif 'tokens' in examples:
        if relation_representation.startswith('EM'):
            token_key = 'tokens_with_typed_marker' if use_entity_typed_marker else 'tokens_with_marker'
        else:
            token_key = 'tokens'
        
        tokenized_inputs = tokenizer(
            examples[token_key],
            padding=padding,
            truncation=True,
            max_length=max_seq_length,
            # We use this argument because the texts in our dataset are lists of words.
            is_split_into_words=True,	
        )
    else:
        raise Exception("There is no tokens element in the data!!")

    labels = []
    relations = []
    predicates = []
    entity_types = []
    
    tokens_seq = []
    tokens_to_ignore = []

    # Most data has a single relation per example, but some data such as SciERC has multiple relations in a sentence.
    for i, rel_list in enumerate(examples['relation']):
        label_ids = []
        relation_spans = []
        predicate_spans = []
        ent_types = []
        
        for rel in rel_list:
            if 'text' in examples:
                
                # ref: https://www.lighttag.io/blog/sequence-labeling-with-transformers/example
                # ref: https://github.com/huggingface/transformers/issues/9326
                def get_token_idx(char_idx):
                    while True:
                        # if it's the last index, return the last token.
                        if char_idx == len(examples[token_key][i]):
                            return len(tokenized_inputs[i]) - 1
                        
                        token_idx = tokenized_inputs.char_to_token(batch_or_char_index=i, char_index=char_idx)
                        # Whitespaces have no token and will return None.
                        if token_idx is not None:
                            return token_idx
                        
                        char_idx += 1
                        # debug
                        #if char_idx == len(examples[token_key][i]):
                        #	raise Exception("End token not found: " f"{tokenizer.convert_ids_to_tokens(tokenized_inputs['input_ids'][i])}")

                e1_span_idx_list, e2_span_idx_list = [], []
                
                if relation_representation.startswith('EM'):
                    if use_entity_typed_marker:
                        e1_idx = rel['entity_1_idx_in_text_with_typed_entity_marker']
                        e2_idx = rel['entity_2_idx_in_text_with_typed_entity_marker']
                    else:
                        e1_idx = rel['entity_1_idx_in_text_with_entity_marker']
                        e2_idx = rel['entity_2_idx_in_text_with_entity_marker']
                    
                    ## TODO: remove this!! the first token is used for separate tokens.
                    if np.asarray(e1_idx).ndim > 1 or np.asarray(e2_idx).ndim > 1:
                        raise Exception("For now, entity marker representations do not support separate entities.")
                else:
                    e1_idx = rel['entity_1_idx']
                    e2_idx = rel['entity_2_idx']
                
                # Some dataset (e.g., BioInfer) has entities consisting of separate tokens.
                # To match the dimension to separate entities, add a dimension for single entities. 
                e1_idx = [e1_idx] if np.asarray(e1_idx).ndim == 1 else e1_idx
                e2_idx = [e2_idx] if np.asarray(e2_idx).ndim == 1 else e2_idx
                
                for e1_s, e1_e in e1_idx:
                    e1_span_s = get_token_idx(e1_s)
                    e1_span_e = get_token_idx(e1_e)
                    e1_span_idx_list.append((e1_span_s, e1_span_e))
                
                for e2_s, e2_e in e2_idx:
                    e2_span_s = get_token_idx(e2_s)
                    e2_span_e = get_token_idx(e2_e)
                    e2_span_idx_list.append((e2_span_s, e2_span_e))
            
            entity_1_type_id = rel['entity_1_type_id']
            entity_2_type_id = rel['entity_2_type_id']

            label_ids.append(rel['relation_id'])
            relation_spans.extend([e1_span_idx_list, e2_span_idx_list])
            ent_types.extend([entity_1_type_id, entity_2_type_id])
            
        labels.append(label_ids)
        relations.append(relation_spans)
        entity_types.append(ent_types)

        if use_context == "attn_based":
            input_tokens = tokenized_inputs.tokens(batch_index=i)
            
            entity_indice = []
            for r_s in relation_spans:
                for span_s, span_e in r_s:
                    entity_indice.extend(list(range(span_s, span_e)))
            
            entity_indice = list(set(entity_indice))

            tokens_seq.append([1 if tok.startswith('##') else 0 for tok in input_tokens])
            tokens_to_ignore.append([-100 if re.search('[a-zA-Z0-9]', tok) == None or \
                                             #tok in tokenizer.all_special_tokens or \
                                             tok in list(set(tokenizer.all_special_tokens) - set(tokenizer.additional_special_tokens)) or \
                                             idx in entity_indice \
                                          else 0 for idx, tok in enumerate(input_tokens)])
            
            tokenized_inputs['tokens_seq'] = tokens_seq
            tokenized_inputs['tokens_to_ignore'] = tokens_to_ignore
 
    
    tokenized_inputs['labels'] = labels
    tokenized_inputs['relations'] = relations
    tokenized_inputs['entity_types'] = entity_types
    
    return tokenized_inputs


def featurize_data(dataset, tokenizer, padding, max_seq_length, relation_representation, use_entity_typed_marker, use_context):
    convert_func_dict = tokenize_and_set_relation_labels
    
    if use_context == "attn_based":
        columns = ['input_ids', 'attention_mask', 'labels', 'token_type_ids', 'relations', 'entity_types', 'tokens_seq', 'tokens_to_ignore']
    else:
        columns = ['input_ids', 'attention_mask', 'labels', 'token_type_ids', 'relations', 'entity_types']
        
    # RoBERTa doesn't use NSP, so it doesn't need 'token_type_ids' which is segement IDs.
    if isinstance(tokenizer, RobertaTokenizerFast):
        columns.remove('token_type_ids')

    features = {}
    for phase, phase_dataset in dataset.items():
        features[phase] = phase_dataset.map(
            convert_func_dict,
            fn_kwargs={'tokenizer': tokenizer,
                       'padding': padding,
                       'max_seq_length': max_seq_length,
                       'relation_representation': relation_representation,
                       'use_entity_typed_marker': use_entity_typed_marker,
                       'use_context': use_context},
            batched=True,
            load_from_cache_file=False,
        )
        
        features[phase].set_format(
            #type="torch",
            type=None,
            columns=columns,
        )
    
    return features

