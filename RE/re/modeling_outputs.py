"""

The argument descriptions come from the class 'SequenceClassifierOutput' in modeling_outputs.py.

"""
import torch
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class RelationClassifierOutput(ModelOutput):
	"""
	Base class for outputs of relation classification models.
	
	Args:
		loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
			Classification (or regression if config.num_labels==1) loss.
		logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
			Classification (or regression if config.num_labels==1) scores (before SoftMax).
		hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
			Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
			of shape :obj:`(batch_size, sequence_length, hidden_size)`.

			Hidden-states of the model at the output of each layer plus the initial embedding outputs.
		attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
			Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
			sequence_length, sequence_length)`.

			Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
			heads.
	"""

	loss: Optional[torch.FloatTensor] = None
	logits: torch.FloatTensor = None
	hidden_states: Optional[Tuple[torch.FloatTensor]] = None
	attentions: Optional[Tuple[torch.FloatTensor]] = None
	
	### TODO: this is used in Training in ver 4.12.0. remove this after test.
	# [START][GP] - return ppi_labels for PPI classification in the joint learning. 10-01-2021
	ppi_labels: torch.FloatTensor = None
	# [END][GP] - return ppi_labels for PPI classification in the joint learning. 10-01-2021
	