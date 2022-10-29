"""

This code referred to the class DataCollatorWithPadding, DataCollatorForTokenClassification in HF data_collator.py

"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers import PreTrainedTokenizerBase


@dataclass
class DataCollatorForRelationClassification:
    
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        import torch
        
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
                
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
            #return_tensors=self.return_tensors,
        )

        label_max_length = max(map(len, batch["labels"]))
        
        
        # [START][GP] - padding 'relations' for relation classification
        if "relations" in batch:
            ## TODO: this will be needed when multiple relations in a single input are supported. 04-21-2022
            #rel_max_length = max(map(len, batch["relations"]))
            
            e1_max_length = max(map(len, [x[0] for x in batch["relations"]]))
            e2_max_length = max(map(len, [x[1] for x in batch["relations"]]))
            e_max_length = max(e1_max_length, e2_max_length)
        # [END][GP] - padding 'relations' for relation classification
        
        # [START][GP] - padding 'predicates' for relation classification. 11-14-2021
        if "predicates" in batch:
            predicate_max_length = max(map(len, batch["predicates"]))
        # [END][GP] - padding 'predicates' for relation classification 11-14-2021
        
        # [START][GP] - padding 'entity_types' for relation classification. 11-23-2021
        if "entity_types" in batch:
            entity_types_max_length = max(map(len, batch["entity_types"]))
        # [END][GP] - padding 'entity_types' for relation classification 11-23-2021
        
        
        # [START][GP] - padding 'tokens_seq' for relation classification. 05-03-2022
        if "tokens_seq" in batch:
            tokens_seq_max_length = max(map(len, batch["tokens_seq"]))
        # [END][GP] - padding 'tokens_seq' for relation classification 05-03-2022


        # [START][GP] - padding 'tokens_to_ignore' for relation classification. 05-03-2022
        if "tokens_to_ignore" in batch:
            tokens_to_ignore_max_length = max(map(len, batch["tokens_to_ignore"]))
        # [END][GP] - padding 'tokens_to_ignore' for relation classification 05-03-2022


        # [START][GP] - padding 'token_seq_idx_with_token_to_ignore_idx' for relation classification. 05-03-2022
        #if "token_seq_idx_with_token_to_ignore_idx" in batch:
        #    token_seq_idx_with_token_to_ignore_idx_max_length = max(map(len, batch["token_seq_idx_with_token_to_ignore_idx"]))
        # [END][GP] - padding 'token_seq_idx_with_token_to_ignore_idx' for relation classification 05-03-2022

        # [START][GP] - padding 'input_tokens' for relation classification. 05-02-2022
        #if "input_tokens" in batch:
        #    tokens_max_length = max(map(len, batch["input_tokens"]))
        # [END][GP] - padding 'input_tokens' for relation classification. 05-02-2022

        # [START][GP] - padding 'input_tokens' for relation classification. 05-02-2022
        #if "test_ids" in batch:
        #    test_ids_max_length = max(map(len, batch["test_ids"]))
        # [END][GP] - padding 'input_tokens' for relation classification. 05-02-2022


        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch[label_name] = [
                list(label) + [self.label_pad_token_id] * (label_max_length - len(label)) for label in labels
            ]
            
            # [START][GP] - padding 'relations' for relation classification
            if "relations" in batch:
                
                ## TODO: this will be needed when multiple relations in a single input are supported. 04-21-2022
                #batch["relations"] = [relation + [self.label_pad_token_id] * (rel_max_length - len(relation)) for relation in batch["relations"]]
                
                for x in batch["relations"]:
                    x[0] = x[0] + [[self.label_pad_token_id, self.label_pad_token_id]] * (e_max_length - len(x[0]))
                    x[1] = x[1] + [[self.label_pad_token_id, self.label_pad_token_id]] * (e_max_length - len(x[1]))
            # [END][GP] - padding 'relations' for relation classification
            
            # [START][GP] - padding 'predicates' for relation classification. 11-14-2021
            if "predicates" in batch:
                batch["predicates"] = [predicate + [self.label_pad_token_id] * (predicate_max_length - len(predicate)) for predicate in batch["predicates"]]
            # [END][GP] - padding 'predicates' for relation classification 11-14-2021
            
            # [START][GP] - padding 'entity_types' for relation classification. 11-14-2021
            if "entity_types" in batch:
                batch["entity_types"] = [entity_type + [self.label_pad_token_id] * (entity_types_max_length - len(entity_type)) for entity_type in batch["entity_types"]]
            # [END][GP] - padding 'entity_types' for relation classification 11-14-2021
            
            # [START][GP] - padding 'tokens_seq' for relation classification. 05-03-2022
            if "tokens_seq" in batch:
                batch["tokens_seq"] = [tokens_seq + [self.label_pad_token_id] * (tokens_seq_max_length - len(tokens_seq)) for tokens_seq in batch["tokens_seq"]]
            # [END][GP] - padding 'tokens_seq' for relation classification 05-03-2022
           
            # [START][GP] - padding 'tokens_to_ignore' for relation classification. 05-03-2022
            if "tokens_to_ignore" in batch:
                batch["tokens_to_ignore"] = [tokens_to_ignore + [self.label_pad_token_id] * (tokens_to_ignore_max_length - len(tokens_to_ignore)) for tokens_to_ignore in batch["tokens_to_ignore"]]
            # [END][GP] - padding 'tokens_to_ignore' for relation classification 05-03-2022
           
            # [START][GP] - padding 'token_seq_idx_with_token_to_ignore_idx' for relation classification. 05-03-2022
            #if "token_seq_idx_with_token_to_ignore_idx" in batch:
            #    batch["token_seq_idx_with_token_to_ignore_idx"] = [token_seq_idx_with_token_to_ignore_idx + [self.label_pad_token_id] * (token_seq_idx_with_token_to_ignore_idx_max_length - len(token_seq_idx_with_token_to_ignore_idx)) for token_seq_idx_with_token_to_ignore_idx in batch["token_seq_idx_with_token_to_ignore_idx"]]
            # [END][GP] - padding 'token_seq_idx_with_token_to_ignore_idx' for relation classification 05-03-2022
            
            # [START][GP] - padding 'input_tokens' for relation classification. 05-02-2022
            #if "input_tokens" in batch:
            #    batch["input_tokens"] = [tokens + [self.tokenizer._pad_token] * (tokens_max_length - len(tokens)) for tokens in batch["input_tokens"]]
            # [END][GP] - padding 'input_tokens' for relation classification. 05-02-2022
            
            # [START][GP] - padding 'input_tokens' for relation classification. 05-02-2022
            #if "test_ids" in batch:
            #    batch["test_ids"] = [tokens + [self.label_pad_token_id] * (test_ids_max_length - len(tokens)) for tokens in batch["test_ids"]]
            # [END][GP] - padding 'input_tokens' for relation classification. 05-02-2022
          
          
        else:
            ### TODO: handle this case.
            pass

        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        # [START][GP] - padding 'input_tokens' for relation classification. 05-02-2022
        #import numpy as np
        #batch = {k: torch.tensor([torch.tensor(x, dtype=torch.int64) for x in v[0]]) if k == "input_tokens" else torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        #batch = {k: torch.from_numpy(np.array(map(float, v[0]))) if k == "input_tokens" else torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        #batch = {k: v if k == "input_tokens" else torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        # [END][GP] - padding 'input_tokens' for relation classification. 05-02-2022

        return batch
