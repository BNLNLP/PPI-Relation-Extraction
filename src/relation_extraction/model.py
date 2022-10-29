import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from transformers import (
    BertModel, 
    BertPreTrainedModel,
    RobertaModel,
    RobertaPreTrainedModel,
)

from torch.nn import MSELoss, CrossEntropyLoss

from modeling_outputs import RelationClassifierOutput


class BertForRelationClassification(BertPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        self.hidden_size = config.hidden_size
        self.finetuning_task = config.finetuning_task
        
        self.relation_representation = kwargs['relation_representation']
        self.use_context = kwargs['use_context'] 
        self.num_entity_types = kwargs['num_entity_types']
        
        self.enable_predicate_span = False
        
        self.tokenizer = kwargs['tokenizer']
        
        if self.relation_representation in ['STANDARD_mention_pooling', 'EM_mention_pooling', 'EM_entity_start']:
            # double sized input for prediction head for RE task since it concats two embeddings. 04-04-2021
            pred_head_input_size = 2
        else:
            pred_head_input_size = 1
        
        if self.use_context:
            pred_head_input_size += 1
        
        self.classifier = nn.Linear(config.hidden_size*pred_head_input_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()
    
    '''
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    '''
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        

        relations=None,
        entity_types=None,
        predicates=None,
        tokens_seq=None,
        tokens_to_ignore=None,

        directed=None,
        reverse=None,
            
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs.pooler_output # outputs[1]
        sequence_output = outputs.last_hidden_state # outputs[0]
        sequence_output = self.dropout(sequence_output)

        if self.config.output_attentions:
            attention_output = outputs.attentions # attention_probs
            attention_output = attention_output[-1] # last layer
            
        # offset used to find local context. In case of entity markers, ignore marker tokens for local context.
        # E.g., in the sample [E1] gene1 [/E1] activates [E2] gene2 [/E2], the local context should be just 'activates'.
        lc_offset = 1 if self.relation_representation.startswith('EM') else 0
        
        # This code is merged in the following logic to make CLS tokens work with other features such context and entity type embeds. 05/05/2022
        '''
        ### TODO: fix the error that 'STANDARD_cls_token', 'EM_cls_token' have the same results.
        if self.relation_representation in ['STANDARD_cls_token', 'EM_cls_token']:
            cls_token = self.dropout(pooled_output)
            #cls_token = sequence_output[:, 0, :]
            logits = self.classifier(cls_token)
        
        else:
        '''
        # This is used for [CLS] token.
        pooled_output_dropout = self.dropout(pooled_output)
        
        buffer = []
        #n = 0 # num_newly_added_label (used for undirected (symmetric) relations)
        
        # iterate batch & collect
        for i in range(sequence_output.size()[0]):
            rel_list = [x for x in torch.split(relations[i], 2) if all(xx == -100 for xx in x.tolist()) is False] # e1_span_idx_list, e2_span_idx_list
            entity_type_list = [x for x in torch.split(entity_types[i], 2) if -100 not in x]

            # In case of EM, a sample has a single relation. In case of marker-free, a sample can have multiple relations.
            #for rel, predicate, entity_type in zip(rel_list, predicate_list, entity_type_list):
            for rel, entity_type in zip(rel_list, entity_type_list):
                e1_span_idx_list = rel[0]
                e2_span_idx_list = rel[1]
                
                # Delete pad index [-100, -100].
                e1_span_idx_list = e1_span_idx_list[e1_span_idx_list.sum(dim=1) > 0]
                e2_span_idx_list = e2_span_idx_list[e2_span_idx_list.sum(dim=1) > 0]

                if self.relation_representation in ['STANDARD_cls_token', 'EM_cls_token']:
                    cls_token = pooled_output_dropout[i]
  
                elif self.relation_representation in ['EM_entity_start']:
                    ## TODO: find a better way later.
                    # Get the min start index.
                    e1_start = torch.min(e1_span_idx_list, dim=0)[0][0]
                    e2_start = torch.min(e2_span_idx_list, dim=0)[0][0]
                    
                    e1_rep = sequence_output[i, e1_start-1, :]
                    e2_rep = sequence_output[i, e2_start-1, :]

                elif self.relation_representation in ['STANDARD_mention_pooling', 'EM_mention_pooling']:
                    all_e1_rep = None
                    for e1_start, e1_end in e1_span_idx_list:
                        e1_rep = sequence_output[i, e1_start:e1_end, :]
                        all_e1_rep = torch.cat((all_e1_rep, e1_rep)) if all_e1_rep is not None else e1_rep
                    
                    e1_rep = torch.max(all_e1_rep, dim=0)[0] # max_pooling
                    del all_e1_rep
                    
                    all_e2_rep = None
                    for e2_start, e2_end in e2_span_idx_list:
                        e2_rep = sequence_output[i, e2_start:e2_end, :]
                        all_e2_rep = torch.cat((all_e2_rep, e2_rep)) if all_e2_rep is not None else e2_rep
                    
                    e2_rep = torch.max(all_e2_rep, dim=0)[0] # max_pooling
                    del all_e2_rep

                if self.use_context == 'attn_based': 
                    all_e1_attn = None
                    for e1_start, e1_end in e1_span_idx_list:
                        e1_attn = attention_output[i,:,e1_start:e1_end,:]
                        all_e1_attn = torch.cat((all_e1_attn, e1_attn), dim=1) if all_e1_attn is not None else e1_attn

                    e1_attn = torch.max(all_e1_attn.sum(0), dim=0)[0] # max_pooling
                    del all_e1_attn
                    
                    all_e2_attn = None
                    for e2_start, e2_end in e2_span_idx_list:
                        e2_attn = attention_output[i,:,e2_start:e2_end,:]
                        all_e2_attn = torch.cat((all_e2_attn, e2_attn), dim=1) if all_e2_attn is not None else e2_attn
                    
                    e2_attn = torch.max(all_e2_attn.sum(0), dim=0)[0] # max_pooling
                    del all_e2_attn
                    
                    b = tokens_to_ignore[i] == -100

                    e1_attn[b.nonzero()] = float("-Inf")
                    e2_attn[b.nonzero()] = float("-Inf")

                    num_of_attentive_tokens = torch.round((e1_attn != float("-Inf")).count_nonzero()*0.2)
                    
                    all_contexts = None
                    
                    ctx_tok_cnt = 0
                    for _ in range(num_of_attentive_tokens.int()):
                        e1_e2_attn_most_idx = torch.argmax(torch.add(e1_attn, e2_attn))

                        # check if a token is a part of a split token.
                        if tokens_seq[i][e1_e2_attn_most_idx] == 1 or tokens_seq[i][e1_e2_attn_most_idx+1] == 1:
                                                   
                            def get_index(list, start, reverse=False):
                                step = -1 if reverse else 1
                                for ii, tt in enumerate(list[start::step]):
                                    if tt != 1:
                                        break
                                return start-ii if reverse else start+ii

                            word_s = get_index(tokens_seq[i].tolist(), e1_e2_attn_most_idx, reverse=True) if tokens_seq[i][e1_e2_attn_most_idx] == 1 else e1_e2_attn_most_idx
                            word_e = get_index(tokens_seq[i].tolist(), e1_e2_attn_most_idx+1, reverse=False)
                            
                            context = sequence_output[i, word_s:word_e, :]
                        else:
                            # To match dimension with the case above.
                            word_s = e1_e2_attn_most_idx
                            word_e = e1_e2_attn_most_idx+1
                            context = sequence_output[i, word_s:word_e, :]

                        e1_attn[word_s:word_e] = float("-Inf")
                        e2_attn[word_s:word_e] = float("-Inf")

                        ctx_tok_cnt += (word_e - word_s)
                        
                        if all_contexts is None:
                            all_contexts = context
                        else:
                            all_contexts = torch.cat((all_contexts, context))
                        
                    if all_contexts is None:
                        context = torch.zeros([self.hidden_size], dtype=sequence_output.dtype, device=sequence_output.device)
                    else:
                        context = torch.max(all_contexts, dim=0)[0] # max_pooling
                        del all_contexts
                    
                elif self.use_context == 'local':
                    # Get the min start index and max end index.
                    e1_start = torch.min(e1_span_idx_list, dim=0)[0][0]
                    e1_end = torch.max(e1_span_idx_list, dim=0)[0][1]
                    e2_start = torch.min(e2_span_idx_list, dim=0)[0][0]
                    e2_end = torch.max(e2_span_idx_list, dim=0)[0][1]

                    # if entity 1 appears before entity 2, and there is at least one token exists betweeen them.
                    if e1_end + lc_offset < e2_start - lc_offset:
                        context = sequence_output[i, e1_end+lc_offset:e2_start-lc_offset, :]
                        context = torch.transpose(context, 0, 1)
                        context = torch.max(context, dim=1)[0] # max_pooling
                    # if entity 2 appears before entity 1, and there is at least one token exists betweeen them.
                    elif e2_end + lc_offset < e1_start - lc_offset:
                        context = sequence_output[i, e2_end+lc_offset:e1_start-lc_offset, :]
                        context = torch.transpose(context, 0, 1)
                        context = torch.max(context, dim=1)[0] # max_pooling
                    else:
                        context = torch.zeros([self.hidden_size], dtype=sequence_output.dtype, device=sequence_output.device)
                
                if self.relation_representation in ['STANDARD_cls_token', 'EM_cls_token']:
                    rel_rep = torch.cat((cls_token, context)) if self.use_context else cls_token
                else:
                    rel_rep = torch.cat((e1_rep, context, e2_rep)) if self.use_context else torch.cat((e1_rep, e2_rep))

                buffer.append(rel_rep)

                if self.relation_representation in ['STANDARD_cls_token', 'EM_cls_token']:
                    del cls_token
                else:
                    del e1_rep
                    del e2_rep
                
                if self.use_context:
                    del context
                
        rel_rep = torch.stack([x for x in buffer], dim=0)
        del buffer

        logits = self.classifier(rel_rep)
        
        loss_fct = CrossEntropyLoss()

        loss = loss_fct(logits, labels.squeeze(1))
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return RelationClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )

