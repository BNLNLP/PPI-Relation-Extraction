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
        
        # [START][GP] - padding 'tokens_seq' for relation classification. 05-03-2022
        if "tokens_seq" in batch:
            tokens_seq_max_length = max(map(len, batch["tokens_seq"]))
        # [END][GP] - padding 'tokens_seq' for relation classification 05-03-2022

        # [START][GP] - padding 'tokens_to_ignore' for relation classification. 05-03-2022
        if "tokens_to_ignore" in batch:
            tokens_to_ignore_max_length = max(map(len, batch["tokens_to_ignore"]))
        # [END][GP] - padding 'tokens_to_ignore' for relation classification 05-03-2022

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
             
            # [START][GP] - padding 'tokens_seq' for relation classification. 05-03-2022
            if "tokens_seq" in batch:
                batch["tokens_seq"] = [tokens_seq + [self.label_pad_token_id] * (tokens_seq_max_length - len(tokens_seq)) for tokens_seq in batch["tokens_seq"]]
            # [END][GP] - padding 'tokens_seq' for relation classification 05-03-2022
           
            # [START][GP] - padding 'tokens_to_ignore' for relation classification. 05-03-2022
            if "tokens_to_ignore" in batch:
                batch["tokens_to_ignore"] = [tokens_to_ignore + [self.label_pad_token_id] * (tokens_to_ignore_max_length - len(tokens_to_ignore)) for tokens_to_ignore in batch["tokens_to_ignore"]]
            # [END][GP] - padding 'tokens_to_ignore' for relation classification 05-03-2022
           
        else:
            ### TODO: handle this case.
            pass

        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}

        return batch
