"""

This code referred to the class DataCollatorWithPadding, DataCollatorForTokenClassification in HF data_collator.py

"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers import PreTrainedTokenizerBase


@dataclass
class DataCollatorForRelationClassification:
	
	### TODO: update the description.
	"""
	
	Data collator that will dynamically pad the inputs received.

	Args:
		tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
			The tokenizer used for encoding the data.
		padding (`bool`, `str` or [`~file_utils.PaddingStrategy`], *optional*, defaults to `True`):
			Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
			among:

			- `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
			  if provided).
			- `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
			  acceptable input length for the model if that argument is not provided.
			- `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
			  lengths).
		max_length (`int`, *optional*):
			Maximum length of the returned list and optionally padding length (see above).
		pad_to_multiple_of (`int`, *optional*):
			If set will pad the sequence to a multiple of the provided value.

			This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
			7.5 (Volta).
		return_tensors (`str`):
			The type of Tensor to return. Allowable values are "np", "pt" and "tf".
	"""

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
			rel_max_length = max(map(len, batch["relations"]))
		# [END][GP] - padding 'relations' for relation classification
		
		# [START][GP] - padding 'predicates' for relation classification. 11-14-2021
		if "predicates" in batch:
			predicate_max_length = max(map(len, batch["predicates"]))
		# [END][GP] - padding 'predicates' for relation classification 11-14-2021
		
		# [START][GP] - padding 'entity_types' for relation classification. 11-23-2021
		if "entity_types" in batch:
			entity_types_max_length = max(map(len, batch["entity_types"]))
		# [END][GP] - padding 'entity_types' for relation classification 11-23-2021
		

		padding_side = self.tokenizer.padding_side
		if padding_side == "right":
			batch[label_name] = [
				list(label) + [self.label_pad_token_id] * (label_max_length - len(label)) for label in labels
			]
			

			# [START][GP] - padding 'relations' for relation classification
			if "relations" in batch:
				batch["relations"] = [relation + [self.label_pad_token_id] * (rel_max_length - len(relation)) for relation in batch["relations"]]
			# [END][GP] - padding 'relations' for relation classification
			
			# [START][GP] - padding 'predicates' for relation classification. 11-14-2021
			if "predicates" in batch:
				batch["predicates"] = [predicate + [self.label_pad_token_id] * (predicate_max_length - len(predicate)) for predicate in batch["predicates"]]
			# [END][GP] - padding 'predicates' for relation classification 11-14-2021
			
			# [START][GP] - padding 'entity_types' for relation classification. 11-14-2021
			if "entity_types" in batch:
				batch["entity_types"] = [entity_type + [self.label_pad_token_id] * (entity_types_max_length - len(entity_type)) for entity_type in batch["entity_types"]]
			# [END][GP] - padding 'entity_types' for relation classification 11-14-2021
		else:
			### TODO: handle this case.
			pass
		
		
		
		batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
		
		'''
		for k, v in batch.items():
			print(k, v)
			print(type(k))
			print(type(v))
		input('enter..')
		'''
			
			
		return batch

