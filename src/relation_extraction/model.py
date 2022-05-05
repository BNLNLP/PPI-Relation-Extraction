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
        self.use_entity_type_embeddings = kwargs['use_entity_type_embeddings']
        self.num_entity_types = kwargs['num_entity_types']
        
        
        
        
        self.enable_predicate_span = False
        
        
        self.tokenizer = kwargs['tokenizer']
        
        if self.use_entity_type_embeddings:
            ### TODO: find the best size. 25 is adopted from SpERT.
            #self.entity_type_emb_size = 25
            #self.entity_type_emb_size = config.hidden_size
            
            # Option 1
            # ref: https://ai.stackexchange.com/questions/28564/how-to-determine-the-embedding-size
            # ref: https://www.quora.com/How-do-I-determine-the-number-of-dimensions-for-word-embedding
            # ref: https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html
            #self.entity_type_emb_size = round((self.num_entity_types*2)**0.25)
            
            # Option 2
            # ref: https://ai.stackexchange.com/questions/28564/how-to-determine-the-embedding-size
            # ref: https://books.google.com/books?id=dDwDEAAAQBAJ&pg=PA48&lpg=PA48&dq=%22If+we%E2%80%99re+in+a+hurry,+one+rule+of+thumb+is+to+use+the+fourth+root+of+the+total+number+of+unique+categorical+elements+while+another+is+that+the+embedding+dimension+should+be+approximately+1.6+times+the+square+root+of+the+number+of+unique+elements+in+the+category,+and+no+less+than+600.%22&source=bl&ots=u9MG_ebFR9&sig=ACfU3U1KZID6yQlF89RmcDPWbNmkexamRg&hl=en&sa=X&ved=2ahUKEwisgJz8h_72AhUTkYkEHZ0IB9QQ6AF6BAgCEAM#v=onepage&q=%22If%20we%E2%80%99re%20in%20a%20hurry%2C%20one%20rule%20of%20thumb%20is%20to%20use%20the%20fourth%20root%20of%20the%20total%20number%20of%20unique%20categorical%20elements%20while%20another%20is%20that%20the%20embedding%20dimension%20should%20be%20approximately%201.6%20times%20the%20square%20root%20of%20the%20number%20of%20unique%20elements%20in%20the%20category%2C%20and%20no%20less%20than%20600.%221.6&f=false
            import math
            self.entity_type_emb_size = round(1.6*math.sqrt(self.num_entity_types*2))		
        
            self.entity_type_embeddings = nn.Embedding(self.num_entity_types*2, self.entity_type_emb_size)

            '''
            # an error occurs when the size is 100. 12-06-2021
            # an error occurs when the size is 200 when it's running on CHEMPROT. 12-07-2021
            # an error occurs when the size is 300 when it's running on DDI. 12-07-2021
            self.pos_diff_embeddings = nn.Embedding(100, config.hidden_size)
            '''
        
        
        if self.relation_representation in ['STANDARD_mention_pooling', 'EM_mention_pooling', 'EM_entity_start']:
            # double sized input for prediction head for RE task since it concats two embeddings. 04-04-2021
            pred_head_input_size = 2
        else:
            pred_head_input_size = 1
        
        if self.use_context:
            pred_head_input_size += 1
        
        if self.use_entity_type_embeddings:
            self.classifier = nn.Linear(config.hidden_size*pred_head_input_size + self.entity_type_emb_size*4, config.num_labels)
        else:
            self.classifier = nn.Linear(config.hidden_size*pred_head_input_size, config.num_labels)

        #self.init_weights() # for older versions.
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
        #token_seq_idx_with_token_to_ignore_idx=None,
        #input_tokens=None,
        #test_ids=None,
        
        
        directed=None,
        reverse=None,
            
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # type(outputs) -> <class 'transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions'>
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
        
        #print(torch.equal(outputs.last_hidden_state, outputs[0]))
        #print(torch.equal(outputs.pooler_output, outputs[1]))
        
        #print(len(outputs))
        #print(type(outputs))
        
        
        
        #print('self.config.output_attentions:', self.config.output_attentions)
        
        
        if self.config.output_attentions:
            attention_output = outputs.attentions # attention_probs
            attention_output = attention_output[-1] # last layer
            
            #attention_output = self.dropout(attention_output)
            
            #print(len(attention_output))
            #print(attention_output[0][0][0][0])
            #print(torch.sum(attention_output[0][0][0][0]))
            #print(attention_output[0][0][0][0].shape)
            #input('etner..;')

        # debug
        '''
        inp = torch.cuda.LongTensor([[0,2,0,5]], device=sequence_output.device)
        print(self.bert.embeddings.position_embeddings) # size
        print(self.bert.embeddings.position_embeddings(inp))
        inp = torch.cuda.LongTensor([[0,1]], device=sequence_output.device)
        print(self.bert.embeddings.token_type_embeddings) # size
        print(self.bert.embeddings.token_type_embeddings(inp))
        input('enter...')
        '''
        
        
        # debug
        #test_tokens = [x.split() for x in self.tokenizer.batch_decode(input_ids)]
        #for x in self.tokenizer.batch_decode(input_ids):
        #    print(x)
        #    input('enter..')
        
        
        
        

        # offset used to find local context. In case of entity markers, ignore marker tokens for local context.
        # E.g., in the sample [E1] gene1 [/E1] activates [E2] gene2 [/E2], the local context should be just 'activates'.
        lc_offset = 1 if self.relation_representation.startswith('EM') else 0

        ### TODO: fix the error that 'STANDARD_cls_token', 'EM_cls_token' have the same results.
        if self.relation_representation in ['STANDARD_cls_token', 'EM_cls_token']:
            cls_token = self.dropout(pooled_output)
            #cls_token = sequence_output[:, 0, :]
            logits = self.classifier(cls_token)
        
        else:
            buffer = []
            #n = 0 # num_newly_added_label (used for undirected (symmetric) relations)
            
            # iterate batch & collect
            for i in range(sequence_output.size()[0]):
                rel_list = [x for x in torch.split(relations[i], 2) if all(xx == -100 for xx in x.tolist()) is False] # e1_span_idx_list, e2_span_idx_list
                entity_type_list = [x for x in torch.split(entity_types[i], 2) if -100 not in x]
                
                '''
                predicate_list = []
                p_i = []
                for j in predicates[i]:
                    if j.item() != -100:
                        p_i.append(j.item())
                    if j.item() == 1000000:
                        p_i.remove(1000000)
                        predicate_list.append(p_i)
                        p_i = []
                '''		
                
                # In case of EM, a sample has a single relation. In case of marker-free, a sample can have multiple relations.
                #for rel, predicate, entity_type in zip(rel_list, predicate_list, entity_type_list):
                for rel, entity_type in zip(rel_list, entity_type_list):
                    e1_span_idx_list = rel[0]
                    e2_span_idx_list = rel[1]
                    
                    # Delete pad index [-100, -100].
                    e1_span_idx_list = e1_span_idx_list[e1_span_idx_list.sum(dim=1) > 0]
                    e2_span_idx_list = e2_span_idx_list[e2_span_idx_list.sum(dim=1) > 0]
                    
                    # Another way to delete pad index.
                    #e1_span_idx_list = e1_span_idx_list[e1_span_idx_list.eq(torch.tensor([-100, -100])) == False]
                    #e2_span_idx_list = e2_span_idx_list[e2_span_idx_list.eq(torch.tensor([-100, -100])) == False]

                    if self.relation_representation in ['EM_entity_start']:
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
                        
                        
                        
                        ''' comparison test -> using entities performs better than using entity start markers. 04-29-2022
                        if self.relation_representation in ['EM_entity_start']:
                            # Get the min start index.
                            e1_start = torch.min(e1_span_idx_list, dim=0)[0][0]
                            e1_end = torch.max(e1_span_idx_list, dim=0)[0][1]
                            e2_start = torch.min(e2_span_idx_list, dim=0)[0][0]
                            e2_end = torch.max(e2_span_idx_list, dim=0)[0][1]

                            e1_attn = attention_output[i,:,e1_start-1,:]
                            e2_attn = attention_output[i,:,e2_start-1,:]

                            e1_attn = e1_attn.sum(0)
                            e2_attn = e2_attn.sum(0)
                            
                        else:
                        ''' 
                        all_e1_attn = None
                        for e1_start, e1_end in e1_span_idx_list:
                            e1_attn = attention_output[i,:,e1_start:e1_end,:]
                            
                            # debug
                            #print(e1_attn.size())
                            #print(e1_attn)
                            #input('enter...')
                            
                            all_e1_attn = torch.cat((all_e1_attn, e1_attn), dim=1) if all_e1_attn is not None else e1_attn
                        
                        # debug
                        '''
                        print(all_e1_attn.sum(0).size())
                        print(all_e1_attn.sum(0))
                        print(all_e1_attn.sum(0).sum(0).size())
                        print(all_e1_attn.sum(0).sum(0))
                        tm = torch.max(all_e1_attn.sum(0), dim=0)[0]
                        print(tm.size())
                        print(tm)
                        input('enter...')
                        '''

                        #e1_attn = all_e1_attn.sum(0).sum(0)
                        e1_attn = torch.max(all_e1_attn.sum(0), dim=0)[0] # max_pooling
                        #e1_attn = torch.mean(all_e1_attn.sum(0), dim=0) # mean_pooling
                        del all_e1_attn
                        
                        all_e2_attn = None
                        for e2_start, e2_end in e2_span_idx_list:
                            e2_attn = attention_output[i,:,e2_start:e2_end,:]
                            all_e2_attn = torch.cat((all_e2_attn, e2_attn), dim=1) if all_e2_attn is not None else e2_attn
                        
                        #e2_attn = all_e2_attn.sum(0).sum(0)
                        e2_attn = torch.max(all_e2_attn.sum(0), dim=0)[0] # max_pooling
                        #e2_attn = torch.mean(all_e2_attn.sum(0), dim=0) # mean_pooling
                        del all_e2_attn
                        
                            
                            
                        
                        
                        
                        ### [START] 1st approach
                        """
                        # ref: https://discuss.pytorch.org/t/find-indices-of-a-tensor-satisfying-a-condition/80802
                        #b = input_ids[i] <= 4
                        #tokens_to_ignore = b.nonzero()
                        #print(tokens_to_ignore)

                        input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i])
                        
                        
                        # Ignore special tokens and tokens not having any alphanumeric character.
                        # Also, exclude entities.
                        tokens_to_ignore = [idx for idx, tok in enumerate(input_tokens) 
                                                    if tok in self.tokenizer.all_special_tokens or 
                                                       re.search('[a-zA-Z0-9]', tok) == None or 
                                                       (idx >= e1_start and idx < e1_end) or (idx >= e2_start and idx < e2_end)]

                        '''
                        for x in tokens_to_ignore:
                            if (x >= e1_start and x < e1_end) or (x >= e2_start and x < e2_end):
                                print('x:', x, '/ token:', input_tokens[x])
                                print(e1_start, e1_end)
                                print(e2_start, e2_end)
                                print(input_ids[i])
                                print(input_tokens)
                                input('enter..')
                        '''

                        #target = torch.tensor(tokens_to_ignore)
                        #tmp_input = e1_attn.clone()
                        e1_attn[tokens_to_ignore] = float("-Inf")
                        e2_attn[tokens_to_ignore] = float("-Inf")
                        """
                        ### [END] 1st approach

                        
                        ### [START] 2nd approach
                        """
                        #print(self.tokenizer.all_special_tokens)
                        #print(self.tokenizer.all_special_ids)
                        
                        input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i])

                        # debug
                        #print(self.tokenizer.convert_ids_to_tokens(input_ids[i]))
                        #print(input_tokens[i])
                        #print(bool(self.tokenizer.convert_ids_to_tokens(input_ids[i]) == input_tokens[i]))
                        
                        #print(input_ids[i])
                        #print(test_ids[i])
                        #print(torch.equal(input_ids[i], test_ids[i]))
                        #input('enter..')

                        #tokens_to_ignore = [idx for idx, tok in enumerate(input_tokens) 
                        #                            if tok in self.tokenizer.all_special_tokens or 
                        #                               re.search('[a-zA-Z0-9]', tok) == None]
                        
                        nonalphanumeric_token_ids = [idx for idx, tok in enumerate(input_tokens) if re.search('[a-zA-Z0-9]', tok) == None]
                        special_token_ids = torch.isin(input_ids[i], torch.tensor(self.tokenizer.all_special_ids, dtype=torch.int, device=sequence_output.device))
                        
                        e1_attn[nonalphanumeric_token_ids] = float("-Inf")
                        e1_attn[special_token_ids] = float("-Inf")
                        e1_attn[e1_start:e1_end] = float("-Inf")
                        e1_attn[e2_start:e2_end] = float("-Inf")
                        
                        e2_attn[nonalphanumeric_token_ids] = float("-Inf")
                        e2_attn[special_token_ids] = float("-Inf")
                        e2_attn[e1_start:e1_end] = float("-Inf")
                        e2_attn[e2_start:e2_end] = float("-Inf")

                        '''
                        tttt = torch.isin(input_ids[i], torch.tensor(self.tokenizer.all_special_ids, dtype=torch.int, device=sequence_output.device))
                        
                        
                        tokens_to_ignore = [idx for idx, tok in enumerate(input_tokens) 
                                                    if re.search('[a-zA-Z0-9]', tok) == None]

                        ttt_attn[tokens_to_ignore] = float("-Inf") 
                        ttt_attn[tttt] = float("-Inf") 
                        ttt_attn[e1_start:e1_end] = float("-Inf")
                        ttt_attn[e2_start:e2_end] = float("-Inf")
                        
                        print(e1_attn)
                        print(ttt_attn)
                        print(torch.equal(e1_attn, ttt_attn))
                        
                        
                        print(tttt)
                        
                        print(tttt.nonzero(as_tuple=True)[0])
                        
                        b = input_ids[i] != self.tokenizer.all_special_ids
                        print(b)
                        tokens_to_ignore = b.nonzero()
                        print(tokens_to_ignore)
                        
                        input('enter..')
                        '''

                        #print(test_attn)
                        #print(e1_attn)
                        #print(test_attn.size())
                        #print(e1_attn.size())
                        #print(torch.equal(e1_attn, test_attn))
                        #input('enter...')
                        """
                        ### [END] 2nd approach


                        ### [START] 3rd approach
                        b = tokens_to_ignore[i] == -100

                        e1_attn[b.nonzero()] = float("-Inf")
                        e2_attn[b.nonzero()] = float("-Inf")

                        ### [END] 3rd approach
                        
                        
                        
                        
                        
                        
                        ### TODO: find the appropriate percentage.
                        #num_of_attentive_tokens = round(len([x for x in e1_attn if x != float("-Inf")])*0.2)
                        num_of_attentive_tokens = torch.round((e1_attn != float("-Inf")).count_nonzero()*0.2)
                        
                        '''
                        print(round(len([x for x in e1_attn if x != float("-Inf")])))
                        print(round(len([x for x in e1_attn if x > 0])))
                        print(num_of_attentive_tokens)
                        print(input_tokens)
                        print(e1_attn)
                        input('enter..')
                        '''
                        
                        all_contexts = None
                        
                        ctx_tok_cnt = 0
                        #while ctx_tok_cnt < num_of_attentive_tokens:
                        for _ in range(num_of_attentive_tokens.int()):
                            #e1_attn_most_val = torch.max(e1_attn) # debug
                            #e1_attn_most_idx = torch.argmax(e1_attn) # debug
                            #e2_attn_most_val = torch.max(e2_attn) # debug
                            #e2_attn_most_idx = torch.argmax(e2_attn) # debug
                            #e1_e2_attn_most_val = torch.max(torch.add(e1_attn, e2_attn)) # debug
                            
                            
                            ### OPTION
                            e1_e2_attn_most_idx = torch.argmax(torch.add(e1_attn, e2_attn))
                            #e1_e2_attn_most_idx = torch.argmax(torch.maximum(e1_attn, e2_attn))

                            
                            ### [START] 1st approach
                            """
                            # check if a token is a part of a split token.
                            if input_tokens[e1_e2_attn_most_idx].startswith('##') or input_tokens[e1_e2_attn_most_idx+1].startswith('##'):
                                                       
                                def get_index(list, start, reverse=False):
                                    step = -1 if reverse else 1
                                    for ii, tt in enumerate(list[start::step]):
                                        #print('ii:', ii, '/ tt:', tt)
                                        if not tt.startswith('##'):
                                            break
                                    
                                    #print('start:', start)
                                    #print('ii:', ii)

                                    return start-ii if reverse else start+ii

                                word_s = get_index(input_tokens, e1_e2_attn_most_idx, reverse=True) if input_tokens[e1_e2_attn_most_idx].startswith('##') else e1_e2_attn_most_idx
                                word_e = get_index(input_tokens, e1_e2_attn_most_idx+1, reverse=False)
                                
                                # debug
                                '''
                                #if (input_tokens[e1_e2_attn_most_idx].startswith('##') and input_tokens[e1_e2_attn_most_idx+1].startswith('##')) or \
                                #   (input_tokens[e1_e2_attn_most_idx-1].startswith('##') and input_tokens[e1_e2_attn_most_idx].startswith('##')):
                                print("all_special_tokens:", self.tokenizer.all_special_tokens)
                                print("tokens_to_ignore:", tokens_to_ignore)
                                print("e1_attn:", e1_attn)
                                print("e2_attn:", e2_attn)
                                print("e1_attn.size():", e1_attn.size())
                                print("e2_attn.size():", e2_attn.size())
                                print("input_ids:", input_ids[i])
                                print("input_tokens:", input_tokens)
                                print("e1_text:", input_tokens[e1_start:e1_end], "(" + str(e1_start.item()) + ", " + str(e1_end.item()) + ")")
                                print("e2_text:", input_tokens[e2_start:e2_end], "(" + str(e2_start.item()) + ", " + str(e2_end.item()) + ")")
                                print("input_tokens[e1_attn_most_idx]   :", input_tokens[e1_attn_most_idx], "(prob: " + str(round(e1_attn_most_val.item(), 2)) + ", idx: " + str(e1_attn_most_idx.item()) + ")")
                                print("input_tokens[e2_attn_most_idx]   :", input_tokens[e2_attn_most_idx], "(prob: " + str(round(e2_attn_most_val.item(), 2)) + ", idx: " + str(e2_attn_most_idx.item()) + ")")
                                print("input_tokens[e1_e2_attn_most_idx]:", input_tokens[e1_e2_attn_most_idx], "(prob: " + str(round(e1_e2_attn_most_val.item(), 2)) + ", idx: " + str(e1_e2_attn_most_idx.item()) + ")")
                                print("attn_ctx_text:", input_tokens[word_s:word_e], "(" + str(word_s.item()) + ", " + str(word_e.item()) + ")")
                                input('enter...')
                                '''
                                
                                context = sequence_output[i, word_s:word_e, :]
                                
                                # debug
                                '''
                                print('sequence_output[i, word_s:word_e, :].size():', sequence_output[i, word_s:word_e, :].size())
                                #print(sequence_output[i, word_s, :10])
                                #print(sequence_output[i, word_e-1, :10])
                                #print(context[:10])
                                input('enter..')
                                '''
                                
                                e1_attn[word_s:word_e] = float("-Inf")
                                e2_attn[word_s:word_e] = float("-Inf")
                                
                            else:
                            """    
                            ### [END] 1st approach
                            
                            ### [START] 3rd approach
                            # check if a token is a part of a split token.
                            if tokens_seq[i][e1_e2_attn_most_idx] == 1 or tokens_seq[i][e1_e2_attn_most_idx+1] == 1:
                                                       
                                def get_index(list, start, reverse=False):
                                    step = -1 if reverse else 1
                                    
                                    #print(list)
                                    
                                    for ii, tt in enumerate(list[start::step]):
                                        #print('ii:', ii, '/ tt:', tt)
                                        if tt != 1:
                                            break
                                    
                                    #print('start:', start)
                                    #print('ii:', ii)

                                    return start-ii if reverse else start+ii

                                word_s = get_index(tokens_seq[i].tolist(), e1_e2_attn_most_idx, reverse=True) if tokens_seq[i][e1_e2_attn_most_idx] == 1 else e1_e2_attn_most_idx
                                word_e = get_index(tokens_seq[i].tolist(), e1_e2_attn_most_idx+1, reverse=False)
                                
                                # debug
                                '''
                                #if (input_tokens[e1_e2_attn_most_idx].startswith('##') and input_tokens[e1_e2_attn_most_idx+1].startswith('##')) or \
                                #   (input_tokens[e1_e2_attn_most_idx-1].startswith('##') and input_tokens[e1_e2_attn_most_idx].startswith('##')):
                                print("all_special_tokens:", self.tokenizer.all_special_tokens)
                                print("tokens_to_ignore:", tokens_to_ignore)
                                print("e1_attn:", e1_attn)
                                print("e2_attn:", e2_attn)
                                print("e1_attn.size():", e1_attn.size())
                                print("e2_attn.size():", e2_attn.size())
                                print("input_ids:", input_ids[i])
                                print("input_tokens:", input_tokens)
                                print("e1_text:", input_tokens[e1_start:e1_end], "(" + str(e1_start.item()) + ", " + str(e1_end.item()) + ")")
                                print("e2_text:", input_tokens[e2_start:e2_end], "(" + str(e2_start.item()) + ", " + str(e2_end.item()) + ")")
                                print("input_tokens[e1_attn_most_idx]   :", input_tokens[e1_attn_most_idx], "(prob: " + str(round(e1_attn_most_val.item(), 2)) + ", idx: " + str(e1_attn_most_idx.item()) + ")")
                                print("input_tokens[e2_attn_most_idx]   :", input_tokens[e2_attn_most_idx], "(prob: " + str(round(e2_attn_most_val.item(), 2)) + ", idx: " + str(e2_attn_most_idx.item()) + ")")
                                print("input_tokens[e1_e2_attn_most_idx]:", input_tokens[e1_e2_attn_most_idx], "(prob: " + str(round(e1_e2_attn_most_val.item(), 2)) + ", idx: " + str(e1_e2_attn_most_idx.item()) + ")")
                                print("attn_ctx_text:", input_tokens[word_s:word_e], "(" + str(word_s.item()) + ", " + str(word_e.item()) + ")")
                                input('enter...')
                                '''
                                
                                context = sequence_output[i, word_s:word_e, :]
                                
                                # debug
                                '''
                                print('sequence_output[i, word_s:word_e, :].size():', sequence_output[i, word_s:word_e, :].size())
                                #print(sequence_output[i, word_s, :10])
                                #print(sequence_output[i, word_e-1, :10])
                                #print(context[:10])
                                input('enter..')
                                '''

                            else:
                            ### [END] 3rd approach

                                #context = sequence_output[i, e1_e2_attn_most_idx, :]
                                # To match dimension with the case above.
                                word_s = e1_e2_attn_most_idx
                                word_e = e1_e2_attn_most_idx+1
                                context = sequence_output[i, word_s:word_e, :]
 
                            e1_attn[word_s:word_e] = float("-Inf")
                            e2_attn[word_s:word_e] = float("-Inf")

                            ctx_tok_cnt += (word_e - word_s)
                            
                            
                            #print(ctx_tok_cnt)
                            #print(num_of_attentive_tokens)
                            #input('enter..')

                            if all_contexts is None:
                                all_contexts = context
                            else:
                                all_contexts = torch.cat((all_contexts, context))
                            #del context                        
                        
                        
                        #print('ctx_tok_cnt:', ctx_tok_cnt)
                        #print('num_of_attentive_tokens:', num_of_attentive_tokens)
                        #input('enter..')
                        
                        
                        if all_contexts is None:
                            context = torch.zeros([self.hidden_size], dtype=sequence_output.dtype, device=sequence_output.device)
                        else:
                            context = torch.max(all_contexts, dim=0)[0] # max_pooling
                            del all_contexts
                            

                        # debug
                        '''
                        print("all_special_tokens:", self.tokenizer.all_special_tokens)
                        print("tokens_to_ignore:", tokens_to_ignore)
                        print(e1_attn)
                        print(e2_attn)
                        print(e1_attn.size())
                        print(e2_attn.size())
                        print(e1_attn_most_val, e1_attn_most_idx)
                        print(e2_attn_most_val, e2_attn_most_idx)
                        print(e1_e2_attn_most_val, e1_e2_attn_most_idx)
                    
                        print(input_ids[i])
                        print(input_tokens)
                        print(input_tokens[e1_start])
                        print(input_tokens[e2_start])
                        print(input_tokens[e1_attn_most_idx])
                        print(input_tokens[e2_attn_most_idx])
                        print(input_tokens[e1_e2_attn_most_idx])
                        input('enter...')
                        '''
                        
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
                            #context = self.dropout(pooled_output[i])
                            
                            # debug
                            '''
                            input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i])
                            print(input_tokens)
                            print(input_tokens[e1_start])
                            print(input_tokens[e1_end])
                            print(input_tokens[e2_start])
                            print(input_tokens[e2_end])
                            
                            print(e1_rep.size())
                            print(e2_rep.size())
                            print(pooled_output[i].size())
                            print(context.size())
                            input('enter...')
                            '''
                    
                    
                    
                    
                    
                    
                    """
                    if self.enable_predicate_span:
                        use_predicate_span = predicate[0]
                        
                        if use_predicate_span:
                            p_s_l = zip(*[iter(predicate[1:])]*2)
                            all_predicate_spans = None
                            for predicate_span_s, predicate_span_e in p_s_l:
                                predicate_span_s = torch.tensor(predicate_span_s, dtype=torch.int, device=sequence_output.device)
                                predicate_span_e = torch.tensor(predicate_span_e, dtype=torch.int, device=sequence_output.device)
                                
                                predicate_span = sequence_output[i, predicate_span_s:predicate_span_e, :]
                                
                                '''
                                print(context)
                                print(predicate_span)
                                print(context.shape)
                                print(predicate_span.shape)
                                print(e1_start, e1_end, e2_start, e2_end)
                                print(predicate_span_s, predicate_span_e)
                                #input('enter..')
                                '''
                                if all_predicate_spans is None:
                                    all_predicate_spans = predicate_span
                                else:
                                    all_predicate_spans = torch.cat((all_predicate_spans, predicate_span))
                                
                                del predicate_span

                            predicate_span = torch.max(all_predicate_spans, dim=0)[0] # max_pooling
                            del all_predicate_spans
                        else:
                            predicate_span = torch.zeros([self.hidden_size], dtype=sequence_output.dtype, device=sequence_output.device)
                        
                        '''
                        if self.enable_ffnn_for_rep:
                            mention_predicate_rep = torch.cat((e1_rep, predicate_span))
                            mention_predicate_rep = torch.cat((mention_predicate_rep, e2_rep))
                        
                            mention_predicate_rep_z = self.predicate_layer_norm(self.predicate_act_funct(self.predicate_funct_h1(mention_predicate_rep)))
                            mention_predicate_rep_z = self.dropout(mention_predicate_rep_z)
                            #mention_predicate_rep_z = self.predicate_layer_norm(self.predicate_act_funct(self.predicate_funct_h2(mention_predicate_rep_z)))
                            #mention_predicate_rep_z = self.dropout(mention_predicate_rep_z)
                            mention_predicate_rep_z = self.predicate_funct_o(mention_predicate_rep_z)
                            mention_predicate_rep_z = self.dropout(mention_predicate_rep_z)
                        '''
                    """
                    
                    
                    
                    
                    
                    
                    if self.use_entity_type_embeddings:
                        e1_type_id = entity_type[0]
                        e2_type_id = entity_type[1]
                        
                        e1_type_start_emb = self.entity_type_embeddings(e1_type_id)
                        e1_type_end_emb = self.entity_type_embeddings(self.num_entity_types + e1_type_id)
                        e2_type_start_emb = self.entity_type_embeddings(e2_type_id)
                        e2_type_end_emb = self.entity_type_embeddings(self.num_entity_types + e2_type_id)
                        
                        '''
                        # add entity span positional embeddings.
                        e1_span_s_pos_embed = self.bert.embeddings.position_embeddings(e1_start-1)
                        e1_span_e_pos_embed = self.bert.embeddings.position_embeddings(e1_end)
                        e2_span_s_pos_embed = self.bert.embeddings.position_embeddings(e2_start-1)
                        e2_span_e_pos_embed = self.bert.embeddings.position_embeddings(e2_end)
                        
                        e1_type_start_emb += e1_span_s_pos_embed
                        e1_type_end_emb += e1_span_e_pos_embed
                        e2_type_start_emb += e2_span_s_pos_embed
                        e2_type_end_emb += e2_span_e_pos_embed
                        '''
                    
                    
                    
                    ### TODO: make it cleaner later.
                    if self.enable_predicate_span:
                        
                        #context = predicate_span
                        #del predicate_span
                        
                        #context = torch.stack((context, predicate_span))
                        #context = torch.max(context, dim=0)[0] # max_pooling
                        
                        context = torch.cat((context, predicate_span))

                    ### TODO: move context in front of rep, and make it simpler.
                    ### But, it seems the model performs better when context is positioned in the middle.
                    # it works better when context is positioned in the middle.
                    if self.use_entity_type_embeddings:
                        rel_rep = torch.cat((e1_type_start_emb, e1_rep))
                        rel_rep = torch.cat((rel_rep, e1_type_end_emb))
                        if self.use_context:
                            rel_rep = torch.cat((rel_rep, context))
                        rel_rep = torch.cat((rel_rep, e2_type_start_emb))
                        rel_rep = torch.cat((rel_rep, e2_rep))
                        rel_rep = torch.cat((rel_rep, e2_type_end_emb))
                    else:
                        rel_rep = torch.cat((e1_rep, e2_rep))
                        rel_rep = torch.cat((e1_rep, context, e2_rep)) if self.use_context \
                                  else torch.cat((e1_rep, e2_rep))
                                      
                    
                    #if self.use_context:
                    #	rel_rep = torch.cat((context, rel_rep))
                    
                                      

                    buffer.append(rel_rep)

                    # debug
                    '''
                    input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i])
                    print(input_tokens)
                    print(input_tokens[e1_start])
                    print(input_tokens[e1_end])
                    print(input_tokens[e1_start:e1_end])
                    print(input_tokens[e2_start])
                    print(input_tokens[e2_end])
                    print(input_tokens[e2_start:e2_end])
                    input('enter...')
                    '''

                    del e1_rep
                    del e2_rep
                    if self.use_context:
                        del context
                    if self.use_entity_type_embeddings:
                        del e1_type_start_emb	
                        del e1_type_end_emb	
                        del e2_type_start_emb	
                        del e2_type_end_emb
                    
                    '''
                    if self.training: # if it's training,
                        # For undirected (symmetric) relations, consider both A-B and B-A. 11-05-2021
                        if directed[i] == False:
                            if self.relation_representation == 'EM_entity_start':
                                em = torch.cat((e2_start_marker, e1_start_marker))
                                buffer.append(em)
                            elif self.relation_representation == 'EM_entity_start_plus_context':
                                em_plus_context = torch.cat((e2_start_marker, context))
                                em_plus_context = torch.cat((em_plus_context, e1_start_marker))
                                buffer.append(em_plus_context)
                            
                            first_half = labels[0:i+n,:]
                            second_half = labels[i+n:,:]
                            i_label = torch.unsqueeze(labels[i+n,:], 0)
                            labels = torch.cat([first_half, i_label, second_half], 0)
                            n += 1

                    else:
                        if reverse[i] == False:
                            v1v2 = blankv1v2[i, i, :, :]
                            v1v2 = torch.cat((v1v2[0], v1v2[1]))
                            buffer.append(v1v2)
                        else:
                            v2v1 = blankv1v2[i, i, :, :]
                            v2v1 = torch.cat((v2v1[1], v2v1[0]))
                            buffer.append(v2v1)
                    '''	
                    
                    '''
                    for idx in input_ids:
                        print('input_ids:', idx)
                    for idx, elem in enumerate(input_ids):	
                        print('input:', self.tokenizer.convert_ids_to_tokens(elem))
                    print('input_ids.size():', input_ids.size())
                    print('e1_start_em_idx:', e1_start_em_idx, '/ e2_start_em_idx:', e2_start_em_idx)
                    print('e1_start:', e1_start, '/ e1_end:', e1_end, '/ e2_start:', e2_start, '/ e2_end:', e2_end)
                    print('e1_start_marker.shape:', e1_start_marker.shape)
                    print('e2_start_marker.shape:', e2_start_marker.shape)
                    print('context.shape:', context.shape)
                    print('em_plus_context.shape:', em_plus_context.shape)
                    input('enter..')
                    '''
                    
            rel_rep = torch.stack([x for x in buffer], dim=0)
            del buffer
            
            
            
            
            
            #rel_rep = self.dropout(rel_rep)
            
            
            
            
            logits = self.classifier(rel_rep)
            
            '''	
            z = self.LayerNorm(self.rep_act_funct(self.rep_funct_h1(rel_rep)))
            #z = self.LayerNorm(self.rep_act_funct(self.rep_funct_h2(z)))
            z = self.rep_funct_o(z)
            z = self.dropout(z)

            logits = self.classifier(z)
            '''	
            
            '''
            z = self.LayerNorm(self.rep_act_funct(self.rep_funct_h1(rel_rep)))
            z = self.dropout(z)
            z = self.LayerNorm(self.rep_act_funct(self.rep_funct_h2(z)))
            z = self.dropout(z)
            z = self.rep_funct_o(z)
            z = self.dropout(z)

            logits = self.classifier(z)
            '''


        loss_fct = CrossEntropyLoss()

        #loss = loss_fct(logits, labels.squeeze(1)[:,0]) # without paddings
        loss = loss_fct(logits, labels.squeeze(1))
        
        '''
        print(logits)
        print(labels)
        print(labels.squeeze(1))
        print(loss)
        print(logits.shape)
        print(labels.shape)
        print(labels.squeeze(1).shape)
        print(loss.shape)
        input('etner...')
        '''
        
        #loss = loss*self.ppi_weight
        #loss = loss/args.gradient_acc_steps
        
        # debug
        '''	
        for i in input_ids:
            print('input_ids:', i)
        for i in sequence_output:
            print('sequence_output:', i)
        print('e1_e2_start:', e1_e2_start)
        print('v1v2:', v1v2)
        print('v1v2.shape:', v1v2.shape)
        print('type(input_ids):', type(input_ids))
        print('input_ids.shape:', input_ids.shape)
        print('type(e1_e2_start):', type(e1_e2_start))
        print('e1_e2_start.shape:', e1_e2_start.shape)
        print('type(sequence_output):', type(sequence_output))
        print('sequence_output.shape:', sequence_output.shape)
        print('logits.shape:', logits.shape)
        print('logits:', logits)
        #print('labels.squeeze(1).shape:', labels.squeeze(1).shape)
        #print('labels.squeeze(1):', labels.squeeze(1))
        #print('labels.squeeze(1)[:,0].shape:', labels.squeeze(1)[:,0].shape)
        #print('labels.squeeze(1)[:,0]:', labels.squeeze(1)[:,0])
        input('enter..')
        '''

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return RelationClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            
            # To avoid the following error during evaluation. 03/31/2022
            # RuntimeError: Sizes of tensors must match except in dimension ...
            #attentions=outputs.attentions,
            
            ### TODO: this is used in Training in ver 4.12.0. remove this after test.
            # [START][GP] - return ppi_labels for PPI classification in the joint learning. 10-01-2021
            #ppi_labels=ppi_labels,
            # [END][GP] - return ppi_labels for PPI classification in the joint learning. 10-01-2021
        )


class RobertaPooler(nn.Module):
    """ Ref: RobertaClassificationHead() """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return x
        

class RobertaForRelationClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        
        # pooling layer takes [CLS] token which is originally used for NSP task. 
        # Since RoBERTa doesn't have NSP task, 'add_pooling_layer' is not necessary.
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        
        self.pooler = RobertaPooler(config)

        self.hidden_size = config.hidden_size
        self.finetuning_task = config.finetuning_task
        
        self.relation_representation = kwargs['relation_representation']
        self.use_context = kwargs['use_context']
        self.use_entity_type_embeddings = kwargs['use_entity_type_embeddings']
        self.num_entity_types = kwargs['num_entity_types']

        # used for debugging
        self.tokenizer = kwargs['tokenizer']
        
        if self.use_entity_type_embeddings:
            ### TODO: find the best size. 25 is adopted from SpERT.
            self.entity_type_emb_size = 25
        
            #self.entity_type_embeddings = nn.Embedding(100, config.hidden_size)
            self.entity_type_embeddings = nn.Embedding(100, self.entity_type_emb_size)

            '''
            # an error occurs when the size is 100. 12-06-2021
            # an error occurs when the size is 200 when it's running on CHEMPROT. 12-07-2021
            # an error occurs when the size is 300 when it's running on DDI. 12-07-2021
            self.pos_diff_embeddings = nn.Embedding(100, config.hidden_size)
            '''
        
        
        if self.relation_representation in ['EM_entity_start', 'STANDARD_mention_pooling', 'EM_mention_pooling']:
            # double sized input for prediction head for RE task since it concats two embeddings. 04-04-2021
            pred_head_input_size = 2	
        else:
            pred_head_input_size = 1
        
        if self.use_context:
            pred_head_input_size += 1
        
        if self.use_entity_type_embeddings:
            self.classifier = nn.Linear(config.hidden_size*pred_head_input_size + self.entity_type_emb_size*4, config.num_labels)
        else:
            self.classifier = nn.Linear(config.hidden_size*pred_head_input_size, config.num_labels)

        self.init_weights() # for older versions.
        # Initialize weights and apply final processing
        #self.post_init()
        


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
        
        directed=None,
        reverse=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
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
        sequence_output = outputs[0]
        
        # RoBERTa doesn't have [CLS] token, but it has an equivalent <s> token. So, get a pooled output of <s> token.
        pooled_output = self.pooler(sequence_output)
        
        # debug
        '''
        inp = torch.cuda.LongTensor([[0,2,0,5]], device=sequence_output.device)
        print(self.bert.embeddings.position_embeddings) # size
        print(self.bert.embeddings.position_embeddings(inp))
        inp = torch.cuda.LongTensor([[0,1]], device=sequence_output.device)
        print(self.bert.embeddings.token_type_embeddings) # size
        print(self.bert.embeddings.token_type_embeddings(inp))
        input('enter...')
        '''
        
        # offset used to find local context. In case of entity markers, ignore marker tokens for local context.
        # E.g., in the sample [E1] gene1 [/E1] activates [E2] gene2 [/E2], the local context should be just 'activates'.
        lc_offset = 1 if self.relation_representation.startswith('EM') else 0
        
        
        
        
        
        
        ### TODO: fix the error that 'STANDARD_cls_token', 'EM_cls_token' have the same results.
        if self.relation_representation in ['STANDARD_cls_token', 'EM_cls_token']:
            logits = self.classifier(pooled_output)
        
        else:
            buffer = []
            n = 0 # num_newly_added_label (used for undirected (symmetric) relations)
            # iterate batch & collect
            for i in range(sequence_output.size()[0]):
                rel_list = [x for x in torch.split(relations[i], 5) if -100 not in x] # e1_span_s, e1_span_e, e2_span_s, e2_span_e, rel['rel_id']
                entity_type_list = [x for x in torch.split(entity_types[i], 2) if -100 not in x]

                # In case of EM, a sample has a single relation. In case of marker-free, a sample can have multiple relations.
                for rel, entity_type in zip(rel_list, entity_type_list):
                    e1_start  = rel[0]
                    e1_end    = rel[1]
                    e2_start  = rel[2]
                    e2_end    = rel[3]
                    rel_label = rel[4]

                    if self.relation_representation in ['EM_entity_start']:
                        e1_rep = sequence_output[i, e1_start-1, :]
                        e2_rep = sequence_output[i, e2_start-1, :]
                        
                    elif self.relation_representation in ['STANDARD_mention_pooling', 'EM_mention_pooling']:
                        e1_rep = sequence_output[i, e1_start:e1_end, :]
                        e1_rep = torch.transpose(e1_rep, 0, 1)
                        e1_rep = torch.max(e1_rep, dim=1)[0] # max_pooling
                        
                        e2_rep = sequence_output[i, e2_start:e2_end, :]
                        e2_rep = torch.transpose(e2_rep, 0, 1)
                        e2_rep = torch.max(e2_rep, dim=1)[0] # max_pooling
                    
                    ### TODO: update !!!
                    if self.is_local_context_used:
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
                            #context = torch.zeros([self.hidden_size], dtype=sequence_output.dtype, device=sequence_output.device)
                            context = pooled_output[i]
                            
                            # debug
                            '''
                            input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i])
                            print(input_tokens)
                            print(input_tokens[e1_start])
                            print(input_tokens[e1_end])
                            print(input_tokens[e2_start])
                            print(input_tokens[e2_end])
                            
                            print(e1_rep.size())
                            print(e2_rep.size())
                            print(pooled_output[i].size())
                            print(context.size())
                            input('enter...')
                            '''

                    if self.use_entity_type_embeddings:
                        e1_type_id = entity_type[0]
                        e2_type_id = entity_type[1]
                        
                        e1_type_start_emb = self.entity_type_embeddings(e1_type_id)
                        e1_type_end_emb = self.entity_type_embeddings(self.num_entity_types + e1_type_id)
                        e2_type_start_emb = self.entity_type_embeddings(e2_type_id)
                        e2_type_end_emb = self.entity_type_embeddings(self.num_entity_types + e2_type_id)
                    
                    if self.use_entity_type_embeddings:
                        rel_rep = torch.cat((e1_type_start_emb, e1_rep))
                        rel_rep = torch.cat((rel_rep, e1_type_end_emb))
                        if self.is_local_context_used:
                            rel_rep = torch.cat((rel_rep, context))
                        rel_rep = torch.cat((rel_rep, e2_type_start_emb))
                        rel_rep = torch.cat((rel_rep, e2_rep))
                        rel_rep = torch.cat((rel_rep, e2_type_end_emb))
                    else:
                        rel_rep = torch.cat((e1_rep, context, e2_rep)) if self.is_local_context_used \
                                  else torch.cat((e1_rep, e2_rep))

                    buffer.append(rel_rep)

                    # debug
                    '''
                    input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i])
                    print(input_tokens)
                    print(input_tokens[e1_start])
                    print(input_tokens[e1_end])
                    print(input_tokens[e1_start:e1_end])
                    print(input_tokens[e2_start])
                    print(input_tokens[e2_end])
                    print(input_tokens[e2_start:e2_end])
                    input('enter...')
                    '''

                    del e1_rep
                    del e2_rep
                    if self.is_local_context_used:
                        del context
                    if self.use_entity_type_embeddings:
                        del e1_type_start_emb	
                        del e1_type_end_emb	
                        del e2_type_start_emb	
                        del e2_type_end_emb
                    
                    '''
                    if self.training: # if it's training,
                        # For undirected (symmetric) relations, consider both A-B and B-A. 11-05-2021
                        if directed[i] == False:
                            if self.relation_representation == 'EM_entity_start':
                                em = torch.cat((e2_start_marker, e1_start_marker))
                                buffer.append(em)
                            elif self.relation_representation == 'EM_entity_start_plus_context':
                                em_plus_context = torch.cat((e2_start_marker, context))
                                em_plus_context = torch.cat((em_plus_context, e1_start_marker))
                                buffer.append(em_plus_context)
                            
                            first_half = labels[0:i+n,:]
                            second_half = labels[i+n:,:]
                            i_label = torch.unsqueeze(labels[i+n,:], 0)
                            labels = torch.cat([first_half, i_label, second_half], 0)
                            n += 1

                    else:
                        if reverse[i] == False:
                            v1v2 = blankv1v2[i, i, :, :]
                            v1v2 = torch.cat((v1v2[0], v1v2[1]))
                            buffer.append(v1v2)
                        else:
                            v2v1 = blankv1v2[i, i, :, :]
                            v2v1 = torch.cat((v2v1[1], v2v1[0]))
                            buffer.append(v2v1)
                    '''	
                    
                    '''
                    for idx in input_ids:
                        print('input_ids:', idx)
                    for idx, elem in enumerate(input_ids):	
                        print('input:', self.tokenizer.convert_ids_to_tokens(elem))
                    print('input_ids.size():', input_ids.size())
                    print('e1_start_em_idx:', e1_start_em_idx, '/ e2_start_em_idx:', e2_start_em_idx)
                    print('e1_start:', e1_start, '/ e1_end:', e1_end, '/ e2_start:', e2_start, '/ e2_end:', e2_end)
                    print('e1_start_marker.shape:', e1_start_marker.shape)
                    print('e2_start_marker.shape:', e2_start_marker.shape)
                    print('context.shape:', context.shape)
                    print('em_plus_context.shape:', em_plus_context.shape)
                    input('enter..')
                    '''
                    
            rel_rep = torch.stack([x for x in buffer], dim=0)
            del buffer

            logits = self.classifier(rel_rep)
            
            '''	
            z = self.LayerNorm(self.rep_act_funct(self.rep_funct_h1(rel_rep)))
            #z = self.LayerNorm(self.rep_act_funct(self.rep_funct_h2(z)))
            z = self.rep_funct_o(z)
            z = self.dropout(z)

            logits = self.classifier(z)
            '''	
            
            '''
            z = self.LayerNorm(self.rep_act_funct(self.rep_funct_h1(rel_rep)))
            z = self.dropout(z)
            z = self.LayerNorm(self.rep_act_funct(self.rep_funct_h2(z)))
            z = self.dropout(z)
            z = self.rep_funct_o(z)
            z = self.dropout(z)

            logits = self.classifier(z)
            '''


        loss_fct = CrossEntropyLoss()

        #loss = loss_fct(logits, labels.squeeze(1)[:,0]) # without paddings
        loss = loss_fct(logits, labels.squeeze(1))
        
        '''
        print(logits)
        print(labels)
        print(labels.squeeze(1))
        print(loss)
        print(logits.shape)
        print(labels.shape)
        print(labels.squeeze(1).shape)
        print(loss.shape)
        input('etner...')
        '''
        
        #loss = loss*self.ppi_weight
        #loss = loss/args.gradient_acc_steps
        
        # debug
        '''	
        for i in input_ids:
            print('input_ids:', i)
        for i in sequence_output:
            print('sequence_output:', i)
        print('e1_e2_start:', e1_e2_start)
        print('v1v2:', v1v2)
        print('v1v2.shape:', v1v2.shape)
        print('type(input_ids):', type(input_ids))
        print('input_ids.shape:', input_ids.shape)
        print('type(e1_e2_start):', type(e1_e2_start))
        print('e1_e2_start.shape:', e1_e2_start.shape)
        print('type(sequence_output):', type(sequence_output))
        print('sequence_output.shape:', sequence_output.shape)
        print('logits.shape:', logits.shape)
        print('logits:', logits)
        #print('labels.squeeze(1).shape:', labels.squeeze(1).shape)
        #print('labels.squeeze(1):', labels.squeeze(1))
        #print('labels.squeeze(1)[:,0].shape:', labels.squeeze(1)[:,0].shape)
        #print('labels.squeeze(1)[:,0]:', labels.squeeze(1)[:,0])
        input('enter..')
        '''

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return RelationClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            
            ### TODO: this is used in Training in ver 4.12.0. remove this after test.
            # [START][GP] - return ppi_labels for PPI classification in the joint learning. 10-01-2021
            #ppi_labels=ppi_labels,
            # [END][GP] - return ppi_labels for PPI classification in the joint learning. 10-01-2021
        )
