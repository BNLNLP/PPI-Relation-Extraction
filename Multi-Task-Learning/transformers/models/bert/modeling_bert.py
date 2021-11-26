# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model. """


import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...file_utils import (
	ModelOutput,
	add_code_sample_docstrings,
	add_start_docstrings,
	add_start_docstrings_to_model_forward,
	replace_return_docstrings,
)
from ...modeling_outputs import (
	BaseModelOutputWithPastAndCrossAttentions,
	BaseModelOutputWithPoolingAndCrossAttentions,
	CausalLMOutputWithCrossAttentions,
	MaskedLMOutput,
	MultipleChoiceModelOutput,
	NextSentencePredictorOutput,
	QuestionAnsweringModelOutput,
	SequenceClassifierOutput,
	TokenClassifierOutput,
)
from ...modeling_utils import (
	PreTrainedModel,
	apply_chunking_to_forward,
	find_pruneable_heads_and_indices,
	prune_linear_layer,
)
from ...utils import logging
from .configuration_bert import BertConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "bert-base-uncased"
_CONFIG_FOR_DOC = "BertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"

BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
	"bert-base-uncased",
	"bert-large-uncased",
	"bert-base-cased",
	"bert-large-cased",
	"bert-base-multilingual-uncased",
	"bert-base-multilingual-cased",
	"bert-base-chinese",
	"bert-base-german-cased",
	"bert-large-uncased-whole-word-masking",
	"bert-large-cased-whole-word-masking",
	"bert-large-uncased-whole-word-masking-finetuned-squad",
	"bert-large-cased-whole-word-masking-finetuned-squad",
	"bert-base-cased-finetuned-mrpc",
	"bert-base-german-dbmdz-cased",
	"bert-base-german-dbmdz-uncased",
	"cl-tohoku/bert-base-japanese",
	"cl-tohoku/bert-base-japanese-whole-word-masking",
	"cl-tohoku/bert-base-japanese-char",
	"cl-tohoku/bert-base-japanese-char-whole-word-masking",
	"TurkuNLP/bert-base-finnish-cased-v1",
	"TurkuNLP/bert-base-finnish-uncased-v1",
	"wietsedv/bert-base-dutch-cased",
	# See all BERT models at https://huggingface.co/models?filter=bert
]


def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
	"""Load tf checkpoints in a pytorch model."""
	try:
		import re

		import numpy as np
		import tensorflow as tf
	except ImportError:
		logger.error(
			"Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
			"https://www.tensorflow.org/install/ for installation instructions."
		)
		raise
	tf_path = os.path.abspath(tf_checkpoint_path)
	logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
	# Load weights from TF model
	init_vars = tf.train.list_variables(tf_path)
	names = []
	arrays = []
	for name, shape in init_vars:
		logger.info(f"Loading TF weight {name} with shape {shape}")
		array = tf.train.load_variable(tf_path, name)
		names.append(name)
		arrays.append(array)

	for name, array in zip(names, arrays):
		name = name.split("/")
		# adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
		# which are not required for using pretrained model
		if any(
			n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
			for n in name
		):
			logger.info(f"Skipping {'/'.join(name)}")
			continue
		pointer = model
		for m_name in name:
			if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
				scope_names = re.split(r"_(\d+)", m_name)
			else:
				scope_names = [m_name]
			if scope_names[0] == "kernel" or scope_names[0] == "gamma":
				pointer = getattr(pointer, "weight")
			elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
				pointer = getattr(pointer, "bias")
			elif scope_names[0] == "output_weights":
				pointer = getattr(pointer, "weight")
			elif scope_names[0] == "squad":
				pointer = getattr(pointer, "classifier")
			else:
				try:
					pointer = getattr(pointer, scope_names[0])
				except AttributeError:
					logger.info(f"Skipping {'/'.join(name)}")
					continue
			if len(scope_names) >= 2:
				num = int(scope_names[1])
				pointer = pointer[num]
		if m_name[-11:] == "_embeddings":
			pointer = getattr(pointer, "weight")
		elif m_name == "kernel":
			array = np.transpose(array)
		try:
			if pointer.shape != array.shape:
				raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")
		except AssertionError as e:
			e.args += (pointer.shape, array.shape)
			raise
		logger.info(f"Initialize PyTorch weight {name}")
		pointer.data = torch.from_numpy(array)
	return model


class BertEmbeddings(nn.Module):
	"""Construct the embeddings from word, position and token_type embeddings."""

	def __init__(self, config):
		super().__init__()
		self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
		self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
		self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

		# self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
		# any TensorFlow checkpoint file
		self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		# position_ids (1, len position emb) is contiguous in memory and exported when serialized
		self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
		self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
		if version.parse(torch.__version__) > version.parse("1.6.0"):
			self.register_buffer(
				"token_type_ids",
				torch.zeros(self.position_ids.size(), dtype=torch.long, device=self.position_ids.device),
				persistent=False,
			)

	def forward(
		self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
	):
		if input_ids is not None:
			input_shape = input_ids.size()
		else:
			input_shape = inputs_embeds.size()[:-1]

		seq_length = input_shape[1]

		if position_ids is None:
			position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

		# Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
		# when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
		# issue #5664
		if token_type_ids is None:
			if hasattr(self, "token_type_ids"):
				buffered_token_type_ids = self.token_type_ids[:, :seq_length]
				buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
				token_type_ids = buffered_token_type_ids_expanded
			else:
				token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

		if inputs_embeds is None:
			inputs_embeds = self.word_embeddings(input_ids)
		token_type_embeddings = self.token_type_embeddings(token_type_ids)

		embeddings = inputs_embeds + token_type_embeddings
		if self.position_embedding_type == "absolute":
			position_embeddings = self.position_embeddings(position_ids)
			embeddings += position_embeddings
		embeddings = self.LayerNorm(embeddings)
		embeddings = self.dropout(embeddings)
		return embeddings


class BertSelfAttention(nn.Module):
	def __init__(self, config):
		super().__init__()
		if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
			raise ValueError(
				f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
				f"heads ({config.num_attention_heads})"
			)

		self.num_attention_heads = config.num_attention_heads
		self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
		self.all_head_size = self.num_attention_heads * self.attention_head_size

		self.query = nn.Linear(config.hidden_size, self.all_head_size)
		self.key = nn.Linear(config.hidden_size, self.all_head_size)
		self.value = nn.Linear(config.hidden_size, self.all_head_size)

		self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
		self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
		if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
			self.max_position_embeddings = config.max_position_embeddings
			self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

		self.is_decoder = config.is_decoder

	def transpose_for_scores(self, x):
		new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
		
		'''
		print('<class BertSelfAttention> transpose_for_scores() x.size()[:-1]:', x.size()[:-1])
		print('<class BertSelfAttention> transpose_for_scores() new_x_shape:', new_x_shape)
		print('<class BertSelfAttention> transpose_for_scores() x.view(*new_x_shape).shape:', x.view(*new_x_shape).shape)
		'''
		
		
		x = x.view(*new_x_shape)
		
		
		'''
		print('<class BertSelfAttention> transpose_for_scores() x.permute(0, 2, 1, 3).shape:', x.permute(0, 2, 1, 3).shape)
		'''
		
		
		
		
		return x.permute(0, 2, 1, 3)

	def forward(
		self,
		hidden_states,
		attention_mask=None,
		head_mask=None,
		encoder_hidden_states=None,
		encoder_attention_mask=None,
		past_key_value=None,
		output_attentions=False,
	):
	
	
	
		#print('<class BertSelfAttention> hidden_states.shape:', hidden_states.shape)
	
	
	
		mixed_query_layer = self.query(hidden_states)

		# If this is instantiated as a cross-attention module, the keys
		# and values come from an encoder; the attention mask needs to be
		# such that the encoder's padding tokens are not attended to.
		is_cross_attention = encoder_hidden_states is not None

		if is_cross_attention and past_key_value is not None:
			# reuse k,v, cross_attentions
			key_layer = past_key_value[0]
			value_layer = past_key_value[1]
			attention_mask = encoder_attention_mask
		elif is_cross_attention:
			key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
			value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
			attention_mask = encoder_attention_mask
		elif past_key_value is not None:
			key_layer = self.transpose_for_scores(self.key(hidden_states))
			value_layer = self.transpose_for_scores(self.value(hidden_states))
			key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
			value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
		else:
			key_layer = self.transpose_for_scores(self.key(hidden_states))
			value_layer = self.transpose_for_scores(self.value(hidden_states))

		query_layer = self.transpose_for_scores(mixed_query_layer)

		if self.is_decoder:
			# if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
			# Further calls to cross_attention layer can then reuse all cross-attention
			# key/value_states (first "if" case)
			# if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
			# all previous decoder key/value_states. Further calls to uni-directional self-attention
			# can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
			# if encoder bi-directional self-attention `past_key_value` is always `None`
			past_key_value = (key_layer, value_layer)

		# Take the dot product between "query" and "key" to get the raw attention scores.
		attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

		if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
			seq_length = hidden_states.size()[1]
			position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
			position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
			distance = position_ids_l - position_ids_r
			positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
			positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

			if self.position_embedding_type == "relative_key":
				relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
				attention_scores = attention_scores + relative_position_scores
			elif self.position_embedding_type == "relative_key_query":
				relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
				relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
				attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

		attention_scores = attention_scores / math.sqrt(self.attention_head_size)
		if attention_mask is not None:
			# Apply the attention mask is (precomputed for all layers in BertModel forward() function)
			attention_scores = attention_scores + attention_mask

		# Normalize the attention scores to probabilities.
		attention_probs = nn.Softmax(dim=-1)(attention_scores)

		# This is actually dropping out entire tokens to attend to, which might
		# seem a bit unusual, but is taken from the original Transformer paper.
		attention_probs = self.dropout(attention_probs)

		# Mask heads if we want to
		if head_mask is not None:
			attention_probs = attention_probs * head_mask

		context_layer = torch.matmul(attention_probs, value_layer)
		
		
				
		
		
		
		'''
		print('<class BertSelfAttention> query_layer.shape:', query_layer.shape)
		print('<class BertSelfAttention> key_layer.shape:', key_layer.shape)
		print('<class BertSelfAttention> key_layer.transpose(-1, -2).shape:', key_layer.transpose(-1, -2).shape)
		print('<class BertSelfAttention> attention_scores.shape:', attention_scores.shape)
		print('<class BertSelfAttention> attention_probs.shape:', attention_probs.shape)
		print('<class BertSelfAttention> value_layer.shape:', value_layer.shape)
		print('<class BertSelfAttention> context_layer.shape:', context_layer.shape)
		input('enter...')
		'''
		
		
		
		
		
		





		context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
		new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
		context_layer = context_layer.view(*new_context_layer_shape)

		outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

		if self.is_decoder:
			outputs = outputs + (past_key_value,)
		return outputs


class BertSelfOutput(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.dense = nn.Linear(config.hidden_size, config.hidden_size)
		self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)

	def forward(self, hidden_states, input_tensor):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.dropout(hidden_states)
		hidden_states = self.LayerNorm(hidden_states + input_tensor)
		return hidden_states


class BertAttention(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.self = BertSelfAttention(config)
		self.output = BertSelfOutput(config)
		self.pruned_heads = set()

	def prune_heads(self, heads):
		if len(heads) == 0:
			return
		heads, index = find_pruneable_heads_and_indices(
			heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
		)

		# Prune linear layers
		self.self.query = prune_linear_layer(self.self.query, index)
		self.self.key = prune_linear_layer(self.self.key, index)
		self.self.value = prune_linear_layer(self.self.value, index)
		self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

		# Update hyper params and store pruned heads
		self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
		self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
		self.pruned_heads = self.pruned_heads.union(heads)

	def forward(
		self,
		hidden_states,
		attention_mask=None,
		head_mask=None,
		encoder_hidden_states=None,
		encoder_attention_mask=None,
		past_key_value=None,
		output_attentions=False,
	):
		self_outputs = self.self(
			hidden_states,
			attention_mask,
			head_mask,
			encoder_hidden_states,
			encoder_attention_mask,
			past_key_value,
			output_attentions,
		)
		attention_output = self.output(self_outputs[0], hidden_states)
		outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
		return outputs


class BertIntermediate(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
		if isinstance(config.hidden_act, str):
			self.intermediate_act_fn = ACT2FN[config.hidden_act]
		else:
			self.intermediate_act_fn = config.hidden_act

	def forward(self, hidden_states):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.intermediate_act_fn(hidden_states)
		return hidden_states


class BertOutput(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
		self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)

	def forward(self, hidden_states, input_tensor):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.dropout(hidden_states)
		hidden_states = self.LayerNorm(hidden_states + input_tensor)
		return hidden_states


class BertLayer(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.chunk_size_feed_forward = config.chunk_size_feed_forward
		self.seq_len_dim = 1
		self.attention = BertAttention(config)
		self.is_decoder = config.is_decoder
		self.add_cross_attention = config.add_cross_attention
		if self.add_cross_attention:
			if not self.is_decoder:
				raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
			self.crossattention = BertAttention(config)
		self.intermediate = BertIntermediate(config)
		self.output = BertOutput(config)

	def forward(
		self,
		hidden_states,
		attention_mask=None,
		head_mask=None,
		encoder_hidden_states=None,
		encoder_attention_mask=None,
		past_key_value=None,
		output_attentions=False,
	):
		# decoder uni-directional self-attention cached key/values tuple is at positions 1,2
		self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
		self_attention_outputs = self.attention(
			hidden_states,
			attention_mask,
			head_mask,
			output_attentions=output_attentions,
			past_key_value=self_attn_past_key_value,
		)
		attention_output = self_attention_outputs[0]

		# if decoder, the last output is tuple of self-attn cache
		if self.is_decoder:
			outputs = self_attention_outputs[1:-1]
			present_key_value = self_attention_outputs[-1]
		else:
			outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

		cross_attn_present_key_value = None
		if self.is_decoder and encoder_hidden_states is not None:
			if not hasattr(self, "crossattention"):
				raise ValueError(
					f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
				)

			# cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
			cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
			cross_attention_outputs = self.crossattention(
				attention_output,
				attention_mask,
				head_mask,
				encoder_hidden_states,
				encoder_attention_mask,
				cross_attn_past_key_value,
				output_attentions,
			)
			attention_output = cross_attention_outputs[0]
			outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

			# add cross-attn cache to positions 3,4 of present_key_value tuple
			cross_attn_present_key_value = cross_attention_outputs[-1]
			present_key_value = present_key_value + cross_attn_present_key_value

		layer_output = apply_chunking_to_forward(
			self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
		)
		outputs = (layer_output,) + outputs

		# if decoder, return the attn key/values as the last output
		if self.is_decoder:
			outputs = outputs + (present_key_value,)

		return outputs

	def feed_forward_chunk(self, attention_output):
		intermediate_output = self.intermediate(attention_output)
		layer_output = self.output(intermediate_output, attention_output)
		return layer_output


class BertEncoder(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config
		self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
		self.gradient_checkpointing = False

	def forward(
		self,
		hidden_states,
		attention_mask=None,
		head_mask=None,
		encoder_hidden_states=None,
		encoder_attention_mask=None,
		past_key_values=None,
		use_cache=None,
		output_attentions=False,
		output_hidden_states=False,
		return_dict=True,
	):
		all_hidden_states = () if output_hidden_states else None
		all_self_attentions = () if output_attentions else None
		all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

		next_decoder_cache = () if use_cache else None
		for i, layer_module in enumerate(self.layer):
			if output_hidden_states:
				all_hidden_states = all_hidden_states + (hidden_states,)

			layer_head_mask = head_mask[i] if head_mask is not None else None
			past_key_value = past_key_values[i] if past_key_values is not None else None

			if self.gradient_checkpointing and self.training:

				if use_cache:
					logger.warning(
						"`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
					)
					use_cache = False

				def create_custom_forward(module):
					def custom_forward(*inputs):
						return module(*inputs, past_key_value, output_attentions)

					return custom_forward

				layer_outputs = torch.utils.checkpoint.checkpoint(
					create_custom_forward(layer_module),
					hidden_states,
					attention_mask,
					layer_head_mask,
					encoder_hidden_states,
					encoder_attention_mask,
				)
			else:
				layer_outputs = layer_module(
					hidden_states,
					attention_mask,
					layer_head_mask,
					encoder_hidden_states,
					encoder_attention_mask,
					past_key_value,
					output_attentions,
				)

			hidden_states = layer_outputs[0]
			if use_cache:
				next_decoder_cache += (layer_outputs[-1],)
			if output_attentions:
				all_self_attentions = all_self_attentions + (layer_outputs[1],)
				if self.config.add_cross_attention:
					all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

		if output_hidden_states:
			all_hidden_states = all_hidden_states + (hidden_states,)

		if not return_dict:
			return tuple(
				v
				for v in [
					hidden_states,
					next_decoder_cache,
					all_hidden_states,
					all_self_attentions,
					all_cross_attentions,
				]
				if v is not None
			)
		return BaseModelOutputWithPastAndCrossAttentions(
			last_hidden_state=hidden_states,
			past_key_values=next_decoder_cache,
			hidden_states=all_hidden_states,
			attentions=all_self_attentions,
			cross_attentions=all_cross_attentions,
		)


class BertPooler(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.dense = nn.Linear(config.hidden_size, config.hidden_size)
		self.activation = nn.Tanh()

	def forward(self, hidden_states):
		# We "pool" the model by simply taking the hidden state corresponding
		# to the first token.
		first_token_tensor = hidden_states[:, 0]
		pooled_output = self.dense(first_token_tensor)
		pooled_output = self.activation(pooled_output)
		return pooled_output


class BertPredictionHeadTransform(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.dense = nn.Linear(config.hidden_size, config.hidden_size)
		if isinstance(config.hidden_act, str):
			self.transform_act_fn = ACT2FN[config.hidden_act]
		else:
			self.transform_act_fn = config.hidden_act
		self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

	def forward(self, hidden_states):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.transform_act_fn(hidden_states)
		hidden_states = self.LayerNorm(hidden_states)
		return hidden_states


class BertLMPredictionHead(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.transform = BertPredictionHeadTransform(config)

		# The output weights are the same as the input embeddings, but there is
		# an output-only bias for each token.
		self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

		self.bias = nn.Parameter(torch.zeros(config.vocab_size))

		# Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
		self.decoder.bias = self.bias

	def forward(self, hidden_states):
		hidden_states = self.transform(hidden_states)
		hidden_states = self.decoder(hidden_states)
		return hidden_states


class BertOnlyMLMHead(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.predictions = BertLMPredictionHead(config)

	def forward(self, sequence_output):
		prediction_scores = self.predictions(sequence_output)
		return prediction_scores


class BertOnlyNSPHead(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.seq_relationship = nn.Linear(config.hidden_size, 2)

	def forward(self, pooled_output):
		seq_relationship_score = self.seq_relationship(pooled_output)
		return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.predictions = BertLMPredictionHead(config)
		self.seq_relationship = nn.Linear(config.hidden_size, 2)

	def forward(self, sequence_output, pooled_output):
		prediction_scores = self.predictions(sequence_output)
		seq_relationship_score = self.seq_relationship(pooled_output)
		return prediction_scores, seq_relationship_score


class BertPreTrainedModel(PreTrainedModel):
	"""
	An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
	models.
	"""

	config_class = BertConfig
	load_tf_weights = load_tf_weights_in_bert
	base_model_prefix = "bert"
	supports_gradient_checkpointing = True
	_keys_to_ignore_on_load_missing = [r"position_ids"]

	def _init_weights(self, module):
		"""Initialize the weights"""
		if isinstance(module, nn.Linear):
			# Slightly different from the TF version which uses truncated_normal for initialization
			# cf https://github.com/pytorch/pytorch/pull/5617
			module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
			if module.bias is not None:
				module.bias.data.zero_()
		elif isinstance(module, nn.Embedding):
			module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
			if module.padding_idx is not None:
				module.weight.data[module.padding_idx].zero_()
		elif isinstance(module, nn.LayerNorm):
			module.bias.data.zero_()
			module.weight.data.fill_(1.0)

	def _set_gradient_checkpointing(self, module, value=False):
		if isinstance(module, BertEncoder):
			module.gradient_checkpointing = value


@dataclass
class BertForPreTrainingOutput(ModelOutput):
	"""
	Output type of :class:`~transformers.BertForPreTraining`.

	Args:
		loss (`optional`, returned when ``labels`` is provided, ``torch.FloatTensor`` of shape :obj:`(1,)`):
			Total loss as the sum of the masked language modeling loss and the next sequence prediction
			(classification) loss.
		prediction_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
			Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
		seq_relationship_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, 2)`):
			Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
			before SoftMax).
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
	prediction_logits: torch.FloatTensor = None
	seq_relationship_logits: torch.FloatTensor = None
	hidden_states: Optional[Tuple[torch.FloatTensor]] = None
	attentions: Optional[Tuple[torch.FloatTensor]] = None


BERT_START_DOCSTRING = r"""

	This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
	methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
	pruning heads etc.)

	This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
	subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
	general usage and behavior.

	Parameters:
		config (:class:`~transformers.BertConfig`): Model configuration class with all the parameters of the model.
			Initializing with a config file does not load the weights associated with the model, only the
			configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
			weights.
"""

BERT_INPUTS_DOCSTRING = r"""
	Args:
		input_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`):
			Indices of input sequence tokens in the vocabulary.

			Indices can be obtained using :class:`~transformers.BertTokenizer`. See
			:meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
			details.

			`What are input IDs? <../glossary.html#input-ids>`__
		attention_mask (:obj:`torch.FloatTensor` of shape :obj:`({0})`, `optional`):
			Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

			- 1 for tokens that are **not masked**,
			- 0 for tokens that are **masked**.

			`What are attention masks? <../glossary.html#attention-mask>`__
		token_type_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
			Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
			1]``:

			- 0 corresponds to a `sentence A` token,
			- 1 corresponds to a `sentence B` token.

			`What are token type IDs? <../glossary.html#token-type-ids>`_
		position_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
			Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
			config.max_position_embeddings - 1]``.

			`What are position IDs? <../glossary.html#position-ids>`_
		head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
			Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:

			- 1 indicates the head is **not masked**,
			- 0 indicates the head is **masked**.

		inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`({0}, hidden_size)`, `optional`):
			Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
			This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
			vectors than the model's internal embedding lookup matrix.
		output_attentions (:obj:`bool`, `optional`):
			Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
			tensors for more detail.
		output_hidden_states (:obj:`bool`, `optional`):
			Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
			more detail.
		return_dict (:obj:`bool`, `optional`):
			Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


@add_start_docstrings(
	"The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",
	BERT_START_DOCSTRING,
)
class BertModel(BertPreTrainedModel):
	"""

	The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
	cross-attention is added between the self-attention layers, following the architecture described in `Attention is
	all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
	Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

	To behave as an decoder the model needs to be initialized with the :obj:`is_decoder` argument of the configuration
	set to :obj:`True`. To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
	argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
	input to the forward pass.
	"""

	def __init__(self, config, add_pooling_layer=True):
		super().__init__(config)
		self.config = config

		self.embeddings = BertEmbeddings(config)
		self.encoder = BertEncoder(config)

		self.pooler = BertPooler(config) if add_pooling_layer else None

		self.init_weights()

	def get_input_embeddings(self):
		return self.embeddings.word_embeddings

	def set_input_embeddings(self, value):
		self.embeddings.word_embeddings = value

	def _prune_heads(self, heads_to_prune):
		"""
		Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
		class PreTrainedModel
		"""
		for layer, heads in heads_to_prune.items():
			self.encoder.layer[layer].attention.prune_heads(heads)

	@add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
	@add_code_sample_docstrings(
		processor_class=_TOKENIZER_FOR_DOC,
		checkpoint=_CHECKPOINT_FOR_DOC,
		output_type=BaseModelOutputWithPoolingAndCrossAttentions,
		config_class=_CONFIG_FOR_DOC,
	)
	def forward(
		self,
		input_ids=None,
		attention_mask=None,
		token_type_ids=None,
		position_ids=None,
		head_mask=None,
		inputs_embeds=None,
		encoder_hidden_states=None,
		encoder_attention_mask=None,
		past_key_values=None,
		use_cache=None,
		output_attentions=None,
		output_hidden_states=None,
		return_dict=None,
	):
		r"""
		encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
			Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
			the model is configured as a decoder.
		encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
			Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
			the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

			- 1 for tokens that are **not masked**,
			- 0 for tokens that are **masked**.
		past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
			Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

			If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
			(those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
			instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
		use_cache (:obj:`bool`, `optional`):
			If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
			decoding (see :obj:`past_key_values`).
		"""
		output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
		output_hidden_states = (
			output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
		)
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		if self.config.is_decoder:
			use_cache = use_cache if use_cache is not None else self.config.use_cache
		else:
			use_cache = False

		if input_ids is not None and inputs_embeds is not None:
			raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
		elif input_ids is not None:
			input_shape = input_ids.size()
		elif inputs_embeds is not None:
			input_shape = inputs_embeds.size()[:-1]
		else:
			raise ValueError("You have to specify either input_ids or inputs_embeds")

		batch_size, seq_length = input_shape
		device = input_ids.device if input_ids is not None else inputs_embeds.device

		# past_key_values_length
		past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

		if attention_mask is None:
			attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

		if token_type_ids is None:
			if hasattr(self.embeddings, "token_type_ids"):
				buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
				buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
				token_type_ids = buffered_token_type_ids_expanded
			else:
				token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

		# We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
		# ourselves in which case we just need to make it broadcastable to all heads.
		extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

		# If a 2D or 3D attention mask is provided for the cross-attention
		# we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
		if self.config.is_decoder and encoder_hidden_states is not None:
			encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
			encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
			if encoder_attention_mask is None:
				encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
			encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
		else:
			encoder_extended_attention_mask = None

		# Prepare head mask if needed
		# 1.0 in head_mask indicate we keep the head
		# attention_probs has shape bsz x n_heads x N x N
		# input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
		# and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
		head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

		embedding_output = self.embeddings(
			input_ids=input_ids,
			position_ids=position_ids,
			token_type_ids=token_type_ids,
			inputs_embeds=inputs_embeds,
			past_key_values_length=past_key_values_length,
		)
		encoder_outputs = self.encoder(
			embedding_output,
			attention_mask=extended_attention_mask,
			head_mask=head_mask,
			encoder_hidden_states=encoder_hidden_states,
			encoder_attention_mask=encoder_extended_attention_mask,
			past_key_values=past_key_values,
			use_cache=use_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)
		sequence_output = encoder_outputs[0]
		pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

		if not return_dict:
			return (sequence_output, pooled_output) + encoder_outputs[1:]

		return BaseModelOutputWithPoolingAndCrossAttentions(
			last_hidden_state=sequence_output,
			pooler_output=pooled_output,
			past_key_values=encoder_outputs.past_key_values,
			hidden_states=encoder_outputs.hidden_states,
			attentions=encoder_outputs.attentions,
			cross_attentions=encoder_outputs.cross_attentions,
		)


@add_start_docstrings(
	"""
	Bert Model with two heads on top as done during the pretraining: a `masked language modeling` head and a `next
	sentence prediction (classification)` head.
	""",
	BERT_START_DOCSTRING,
)
class BertForPreTraining(BertPreTrainedModel):
	def __init__(self, config):
		super().__init__(config)

		self.bert = BertModel(config)
		self.cls = BertPreTrainingHeads(config)

		self.init_weights()

	def get_output_embeddings(self):
		return self.cls.predictions.decoder

	def set_output_embeddings(self, new_embeddings):
		self.cls.predictions.decoder = new_embeddings

	@add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
	@replace_return_docstrings(output_type=BertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
	def forward(
		self,
		input_ids=None,
		attention_mask=None,
		token_type_ids=None,
		position_ids=None,
		head_mask=None,
		inputs_embeds=None,
		labels=None,
		next_sentence_label=None,
		output_attentions=None,
		output_hidden_states=None,
		return_dict=None,
	):
		r"""
		labels (:obj:`torch.LongTensor` of shape ``(batch_size, sequence_length)``, `optional`):
			Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
			config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
			(masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
		next_sentence_label (``torch.LongTensor`` of shape ``(batch_size,)``, `optional`):
			Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
			(see :obj:`input_ids` docstring) Indices should be in ``[0, 1]``:

			- 0 indicates sequence B is a continuation of sequence A,
			- 1 indicates sequence B is a random sequence.
		kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
			Used to hide legacy arguments that have been deprecated.

		Returns:

		Example::

			>>> from transformers import BertTokenizer, BertForPreTraining
			>>> import torch

			>>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
			>>> model = BertForPreTraining.from_pretrained('bert-base-uncased')

			>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
			>>> outputs = model(**inputs)

			>>> prediction_logits = outputs.prediction_logits
			>>> seq_relationship_logits = outputs.seq_relationship_logits
		"""
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

		sequence_output, pooled_output = outputs[:2]
		prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

		total_loss = None
		if labels is not None and next_sentence_label is not None:
			loss_fct = CrossEntropyLoss()
			masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
			next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
			total_loss = masked_lm_loss + next_sentence_loss

		if not return_dict:
			output = (prediction_scores, seq_relationship_score) + outputs[2:]
			return ((total_loss,) + output) if total_loss is not None else output

		return BertForPreTrainingOutput(
			loss=total_loss,
			prediction_logits=prediction_scores,
			seq_relationship_logits=seq_relationship_score,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)


@add_start_docstrings(
	"""Bert Model with a `language modeling` head on top for CLM fine-tuning. """, BERT_START_DOCSTRING
)
class BertLMHeadModel(BertPreTrainedModel):

	_keys_to_ignore_on_load_unexpected = [r"pooler"]
	_keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

	def __init__(self, config):
		super().__init__(config)

		if not config.is_decoder:
			logger.warning("If you want to use `BertLMHeadModel` as a standalone, add `is_decoder=True.`")

		self.bert = BertModel(config, add_pooling_layer=False)
		self.cls = BertOnlyMLMHead(config)

		self.init_weights()

	def get_output_embeddings(self):
		return self.cls.predictions.decoder

	def set_output_embeddings(self, new_embeddings):
		self.cls.predictions.decoder = new_embeddings

	@add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
	@replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
	def forward(
		self,
		input_ids=None,
		attention_mask=None,
		token_type_ids=None,
		position_ids=None,
		head_mask=None,
		inputs_embeds=None,
		encoder_hidden_states=None,
		encoder_attention_mask=None,
		labels=None,
		past_key_values=None,
		use_cache=None,
		output_attentions=None,
		output_hidden_states=None,
		return_dict=None,
	):
		r"""
		encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
			Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
			the model is configured as a decoder.
		encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
			Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
			the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

			- 1 for tokens that are **not masked**,
			- 0 for tokens that are **masked**.
		labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
			Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
			``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are
			ignored (masked), the loss is only computed for the tokens with labels n ``[0, ..., config.vocab_size]``
		past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
			Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

			If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
			(those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
			instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
		use_cache (:obj:`bool`, `optional`):
			If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
			decoding (see :obj:`past_key_values`).

		Returns:

		Example::

			>>> from transformers import BertTokenizer, BertLMHeadModel, BertConfig
			>>> import torch

			>>> tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
			>>> config = BertConfig.from_pretrained("bert-base-cased")
			>>> config.is_decoder = True
			>>> model = BertLMHeadModel.from_pretrained('bert-base-cased', config=config)

			>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
			>>> outputs = model(**inputs)

			>>> prediction_logits = outputs.logits
		"""
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict
		if labels is not None:
			use_cache = False

		outputs = self.bert(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
			position_ids=position_ids,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
			encoder_hidden_states=encoder_hidden_states,
			encoder_attention_mask=encoder_attention_mask,
			past_key_values=past_key_values,
			use_cache=use_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		sequence_output = outputs[0]
		prediction_scores = self.cls(sequence_output)

		lm_loss = None
		if labels is not None:
			# we are doing next-token prediction; shift prediction scores and input ids by one
			shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
			labels = labels[:, 1:].contiguous()
			loss_fct = CrossEntropyLoss()
			lm_loss = loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

		if not return_dict:
			output = (prediction_scores,) + outputs[2:]
			return ((lm_loss,) + output) if lm_loss is not None else output

		return CausalLMOutputWithCrossAttentions(
			loss=lm_loss,
			logits=prediction_scores,
			past_key_values=outputs.past_key_values,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
			cross_attentions=outputs.cross_attentions,
		)

	def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, **model_kwargs):
		input_shape = input_ids.shape
		# if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
		if attention_mask is None:
			attention_mask = input_ids.new_ones(input_shape)

		# cut decoder_input_ids if past is used
		if past is not None:
			input_ids = input_ids[:, -1:]

		return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past}

	def _reorder_cache(self, past, beam_idx):
		reordered_past = ()
		for layer_past in past:
			reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
		return reordered_past


@add_start_docstrings("""Bert Model with a `language modeling` head on top. """, BERT_START_DOCSTRING)
class BertForMaskedLM(BertPreTrainedModel):

	_keys_to_ignore_on_load_unexpected = [r"pooler"]
	_keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

	def __init__(self, config):
		super().__init__(config)

		if config.is_decoder:
			logger.warning(
				"If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
				"bi-directional self-attention."
			)

		self.bert = BertModel(config, add_pooling_layer=False)
		self.cls = BertOnlyMLMHead(config)

		self.init_weights()

	def get_output_embeddings(self):
		return self.cls.predictions.decoder

	def set_output_embeddings(self, new_embeddings):
		self.cls.predictions.decoder = new_embeddings

	@add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
	@add_code_sample_docstrings(
		processor_class=_TOKENIZER_FOR_DOC,
		checkpoint=_CHECKPOINT_FOR_DOC,
		output_type=MaskedLMOutput,
		config_class=_CONFIG_FOR_DOC,
	)
	def forward(
		self,
		input_ids=None,
		attention_mask=None,
		token_type_ids=None,
		position_ids=None,
		head_mask=None,
		inputs_embeds=None,
		encoder_hidden_states=None,
		encoder_attention_mask=None,
		labels=None,
		output_attentions=None,
		output_hidden_states=None,
		return_dict=None,
	):
		r"""
		labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
			Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
			config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
			(masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
		"""

		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		outputs = self.bert(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
			position_ids=position_ids,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
			encoder_hidden_states=encoder_hidden_states,
			encoder_attention_mask=encoder_attention_mask,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		sequence_output = outputs[0]
		prediction_scores = self.cls(sequence_output)

		masked_lm_loss = None
		if labels is not None:
			loss_fct = CrossEntropyLoss()  # -100 index = padding token
			masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

		if not return_dict:
			output = (prediction_scores,) + outputs[2:]
			return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

		return MaskedLMOutput(
			loss=masked_lm_loss,
			logits=prediction_scores,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)

	def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
		input_shape = input_ids.shape
		effective_batch_size = input_shape[0]

		#  add a dummy token
		assert self.config.pad_token_id is not None, "The PAD token should be defined for generation"
		attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
		dummy_token = torch.full(
			(effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
		)
		input_ids = torch.cat([input_ids, dummy_token], dim=1)

		return {"input_ids": input_ids, "attention_mask": attention_mask}


@add_start_docstrings(
	"""Bert Model with a `next sentence prediction (classification)` head on top. """,
	BERT_START_DOCSTRING,
)
class BertForNextSentencePrediction(BertPreTrainedModel):
	def __init__(self, config):
		super().__init__(config)

		self.bert = BertModel(config)
		self.cls = BertOnlyNSPHead(config)

		self.init_weights()

	@add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
	@replace_return_docstrings(output_type=NextSentencePredictorOutput, config_class=_CONFIG_FOR_DOC)
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
		**kwargs,
	):
		r"""
		labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
			Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
			(see ``input_ids`` docstring). Indices should be in ``[0, 1]``:

			- 0 indicates sequence B is a continuation of sequence A,
			- 1 indicates sequence B is a random sequence.

		Returns:

		Example::

			>>> from transformers import BertTokenizer, BertForNextSentencePrediction
			>>> import torch

			>>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
			>>> model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')

			>>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
			>>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
			>>> encoding = tokenizer(prompt, next_sentence, return_tensors='pt')

			>>> outputs = model(**encoding, labels=torch.LongTensor([1]))
			>>> logits = outputs.logits
			>>> assert logits[0, 0] < logits[0, 1] # next sentence was random
		"""

		if "next_sentence_label" in kwargs:
			warnings.warn(
				"The `next_sentence_label` argument is deprecated and will be removed in a future version, use `labels` instead.",
				FutureWarning,
			)
			labels = kwargs.pop("next_sentence_label")

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

		pooled_output = outputs[1]

		seq_relationship_scores = self.cls(pooled_output)

		next_sentence_loss = None
		if labels is not None:
			loss_fct = CrossEntropyLoss()
			next_sentence_loss = loss_fct(seq_relationship_scores.view(-1, 2), labels.view(-1))

		if not return_dict:
			output = (seq_relationship_scores,) + outputs[2:]
			return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output

		return NextSentencePredictorOutput(
			loss=next_sentence_loss,
			logits=seq_relationship_scores,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)


@add_start_docstrings(
	"""
	Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
	output) e.g. for GLUE tasks.
	""",
	BERT_START_DOCSTRING,
)
class BertForSequenceClassification(BertPreTrainedModel):
	def __init__(self, config):
		super().__init__(config)
		self.num_labels = config.num_labels
		self.config = config

		self.bert = BertModel(config)
		classifier_dropout = (
			config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
		)
		self.dropout = nn.Dropout(classifier_dropout)
		self.classifier = nn.Linear(config.hidden_size, config.num_labels)

		self.init_weights()

	@add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
	@add_code_sample_docstrings(
		processor_class=_TOKENIZER_FOR_DOC,
		checkpoint=_CHECKPOINT_FOR_DOC,
		output_type=SequenceClassifierOutput,
		config_class=_CONFIG_FOR_DOC,
	)
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
	):
		r"""
		labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
			Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
			config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
			If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
		"""
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

		pooled_output = outputs[1]

		pooled_output = self.dropout(pooled_output)
		logits = self.classifier(pooled_output)

		loss = None
		if labels is not None:
			if self.config.problem_type is None:
				if self.num_labels == 1:
					self.config.problem_type = "regression"
				elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
					self.config.problem_type = "single_label_classification"
				else:
					self.config.problem_type = "multi_label_classification"

			if self.config.problem_type == "regression":
				loss_fct = MSELoss()
				if self.num_labels == 1:
					loss = loss_fct(logits.squeeze(), labels.squeeze())
				else:
					loss = loss_fct(logits, labels)
			elif self.config.problem_type == "single_label_classification":
				loss_fct = CrossEntropyLoss()
				loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
			elif self.config.problem_type == "multi_label_classification":
				loss_fct = BCEWithLogitsLoss()
				loss = loss_fct(logits, labels)
		if not return_dict:
			output = (logits,) + outputs[2:]
			return ((loss,) + output) if loss is not None else output

		return SequenceClassifierOutput(
			loss=loss,
			logits=logits,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)


@add_start_docstrings(
	"""
	Bert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
	softmax) e.g. for RocStories/SWAG tasks.
	""",
	BERT_START_DOCSTRING,
)
class BertForMultipleChoice(BertPreTrainedModel):
	def __init__(self, config):
		super().__init__(config)

		self.bert = BertModel(config)
		classifier_dropout = (
			config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
		)
		self.dropout = nn.Dropout(classifier_dropout)
		self.classifier = nn.Linear(config.hidden_size, 1)

		self.init_weights()

	@add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
	@add_code_sample_docstrings(
		processor_class=_TOKENIZER_FOR_DOC,
		checkpoint=_CHECKPOINT_FOR_DOC,
		output_type=MultipleChoiceModelOutput,
		config_class=_CONFIG_FOR_DOC,
	)
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
	):
		r"""
		labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
			Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
			num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
			:obj:`input_ids` above)
		"""
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict
		num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

		input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
		attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
		token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
		position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
		inputs_embeds = (
			inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
			if inputs_embeds is not None
			else None
		)

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

		pooled_output = outputs[1]

		pooled_output = self.dropout(pooled_output)
		logits = self.classifier(pooled_output)
		reshaped_logits = logits.view(-1, num_choices)

		loss = None
		if labels is not None:
			loss_fct = CrossEntropyLoss()
			loss = loss_fct(reshaped_logits, labels)

		if not return_dict:
			output = (reshaped_logits,) + outputs[2:]
			return ((loss,) + output) if loss is not None else output

		return MultipleChoiceModelOutput(
			loss=loss,
			logits=reshaped_logits,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)


# [GP][START]
import itertools
import re
# [GP][END]




class PredicateAttention(nn.Module):
	def __init__(self, config):
		super().__init__()
		if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
			raise ValueError(
				f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
				f"heads ({config.num_attention_heads})"
			)
		
		'''
		self.num_attention_heads = config.num_attention_heads
		self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
		self.all_head_size = self.num_attention_heads * self.attention_head_size

		self.query = nn.Linear(config.hidden_size, self.all_head_size)
		self.key = nn.Linear(config.hidden_size, self.all_head_size)
		self.value = nn.Linear(config.hidden_size, self.all_head_size)
		'''
		self.num_attention_heads = config.num_attention_heads
		self.attention_head_size = int(config.hidden_size*2 / config.num_attention_heads)
		self.all_head_size = self.num_attention_heads * self.attention_head_size

		self.query = nn.Linear(config.hidden_size*2, self.all_head_size)
		self.key = nn.Linear(config.hidden_size*2, self.all_head_size)
		self.value = nn.Linear(config.hidden_size*2, self.all_head_size)
		
		
		
		self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
		
		self.is_decoder = config.is_decoder

	def transpose_for_scores(self, x):
		new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
		
		'''
		print('<class PredicateAttention> transpose_for_scores() x.size()[:-1]:', x.size()[:-1])
		print('<class PredicateAttention> transpose_for_scores() new_x_shape:', new_x_shape)
		print('<class PredicateAttention> transpose_for_scores() x.view(*new_x_shape).shape:', x.view(*new_x_shape).shape)
		'''
		
		
		x = x.view(*new_x_shape)
		
		
		'''
		print('<class PredicateAttention> transpose_for_scores() x.permute(1, 0, 2).shape:', x.permute(1, 0, 2).shape)
		#input('enter...')
		'''
		
		
		
		#return x.permute(0, 2, 1, 3)
		return x.permute(1, 0, 2)

	def forward(
		self,
		hidden_states=None,
		attention_mask=None,
		head_mask=None,
		encoder_hidden_states=None,
		encoder_attention_mask=None,
		past_key_value=None,
		output_attentions=False,
		
		
		
		
		concat_e1_e2=None,
		predicates=None,
		
		
	):
	
	
		
				
		
		
		'''
		#print(self.num_attention_heads)
		#print(self.attention_head_size)
		#print(self.all_head_size)
		print('<class PredicateAttention> concat_e1_e2.shape:', concat_e1_e2.shape)
		print('<class PredicateAttention> predicates.shape:', predicates.shape)
		'''
		
		
		
		



	
		mixed_query_layer = self.query(concat_e1_e2)
		
		'''
		# If this is instantiated as a cross-attention module, the keys
		# and values come from an encoder; the attention mask needs to be
		# such that the encoder's padding tokens are not attended to.
		is_cross_attention = encoder_hidden_states is not None

		if is_cross_attention and past_key_value is not None:
			# reuse k,v, cross_attentions
			key_layer = past_key_value[0]
			value_layer = past_key_value[1]
			attention_mask = encoder_attention_mask
		elif is_cross_attention:
			key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
			value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
			attention_mask = encoder_attention_mask
		elif past_key_value is not None:
			key_layer = self.transpose_for_scores(self.key(hidden_states))
			value_layer = self.transpose_for_scores(self.value(hidden_states))
			key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
			value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
		else:
		'''	
		
		key_layer = self.transpose_for_scores(self.key(predicates))
		value_layer = self.transpose_for_scores(self.value(predicates))

		query_layer = self.transpose_for_scores(mixed_query_layer)

		if self.is_decoder:
			# if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
			# Further calls to cross_attention layer can then reuse all cross-attention
			# key/value_states (first "if" case)
			# if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
			# all previous decoder key/value_states. Further calls to uni-directional self-attention
			# can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
			# if encoder bi-directional self-attention `past_key_value` is always `None`
			past_key_value = (key_layer, value_layer)

		# Take the dot product between "query" and "key" to get the raw attention scores.
		attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))


		attention_scores = attention_scores / math.sqrt(self.attention_head_size)
		if attention_mask is not None:
			# Apply the attention mask is (precomputed for all layers in BertModel forward() function)
			attention_scores = attention_scores + attention_mask

		# Normalize the attention scores to probabilities.
		attention_probs = nn.Softmax(dim=-1)(attention_scores)

		# This is actually dropping out entire tokens to attend to, which might
		# seem a bit unusual, but is taken from the original Transformer paper.
		attention_probs = self.dropout(attention_probs)

		# Mask heads if we want to
		if head_mask is not None:
			attention_probs = attention_probs * head_mask

		context_layer = torch.matmul(attention_probs, value_layer)

		#context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
		context_layer = context_layer.permute(1, 0, 2).contiguous()
		
		
		
		
		new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
		context_layer = context_layer.view(*new_context_layer_shape)

		outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

		if self.is_decoder:
			outputs = outputs + (past_key_value,)
		return outputs



class entity_concat_ffnn(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
		if isinstance(config.hidden_act, str):
			self.intermediate_act_fn = ACT2FN[config.hidden_act]
		else:
			self.intermediate_act_fn = config.hidden_act

	def forward(self, hidden_states):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.intermediate_act_fn(hidden_states)
		return hidden_states
		



@add_start_docstrings(
	"""
	Bert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
	Named-Entity-Recognition (NER) tasks.
	""",
	BERT_START_DOCSTRING,
)
class BertForTokenClassification(BertPreTrainedModel):

	_keys_to_ignore_on_load_unexpected = [r"pooler"]

	def __init__(self, config, **kwargs):
		super().__init__(config)
		self.num_labels = config.num_labels

		self.bert = BertModel(config, add_pooling_layer=False)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		
		self.hidden_size = config.hidden_size
		self.finetuning_task = config.finetuning_task

		self.relation_representation = kwargs['relation_representation']
		self.num_ppi_labels = kwargs['num_ppi_labels'] # used for 'joint-ner-ppi' task because config.num_labels are the number of NER labels.
		self.num_entity_types = kwargs['num_entity_types']
		self.tokenizer = kwargs['tokenizer'] # used to find entity span for 'joint-ner-ppi' task.
		
		self.ner_weight = kwargs['task_weights']['ner']
		self.ppi_weight = kwargs['task_weights']['ppi']

		if self.finetuning_task == 'ner':
			self.ner_classifier = nn.Linear(config.hidden_size, config.num_labels)
		elif self.finetuning_task == 'ppi':
			if self.relation_representation in ['STANDARD_mention_pooling', 'EM_entity_start']:
				# double sized input for prediction head for PPI task since it concats two embeddings. 04-04-2021
				self.rel_classifier = nn.Linear(config.hidden_size*2, config.num_labels)
			elif self.relation_representation in ['STANDARD_mention_pooling_plus_context', 'EM_entity_start_plus_context',
												  'POSITIONAL_mention_pooling_plus_context']:
				
				
				
				
				"""
				Caution!! 11-11-2021
				If any nn.Linear is not used, comment them even if they are not used. For some reason, the nn.Linear affects the performance.
				For instance, when self.span_classifier is alive even though it's not used, the results are different. 
				"""
				self.enable_mention_pooling_and_context = True
				self.enable_predicate_span = False
				
				
				self.enable_entity_type_emb = True
				
				
				self.enable_predicate_attention = False
				
				
				self.enable_pos_emb_loss = False
				self.enable_ffnn_for_rep = False
				
				if self.enable_mention_pooling_and_context:
					#self.span_width_embeddings = nn.Embedding(100, config.hidden_size)
					
					
					
					if self.enable_predicate_attention:
						self.predicate_attention = PredicateAttention(config)
						
						#self.ent_concat_ffnn = entity_concat_ffnn(config)
						
						
						num_of_concat_mention_span_embeds = 2
						context_funct_hidden_size = num_of_concat_mention_span_embeds*4
						self.context_funct_h1 = nn.Linear(config.hidden_size*num_of_concat_mention_span_embeds, config.hidden_size*context_funct_hidden_size)
						self.context_funct_h2 = nn.Linear(config.hidden_size*context_funct_hidden_size, config.hidden_size*context_funct_hidden_size)
						self.context_funct_o = nn.Linear(config.hidden_size*context_funct_hidden_size, config.hidden_size)
						self.context_act_funct = nn.GELU()
						self.context_layer_norm = nn.LayerNorm(config.hidden_size*context_funct_hidden_size, eps=config.layer_norm_eps)
						
					
					
					
					
					if self.enable_ffnn_for_rep:
						# TODO: make it cleaner.
						if (self.enable_predicate_span and self.enable_pos_emb_loss == False) or \
						   (self.enable_predicate_span == False and self.enable_pos_emb_loss):
							num_of_concat_embeds = 2
						else:
							num_of_concat_embeds = 3
					else:
						# TODO: complete this.
						num_of_concat_embeds = 7

					self.rel_classifier = nn.Linear(config.hidden_size*num_of_concat_embeds, config.num_labels)
				
				
				
				if self.enable_entity_type_emb:
					self.entity_type_embeddings = nn.Embedding(100, config.hidden_size)
				
				
				
				
				if self.enable_pos_emb_loss:
					self.pos_diff_embeddings = nn.Embedding(1000, config.hidden_size)
				
				if self.enable_ffnn_for_rep:
					if self.enable_predicate_span or self.enable_pos_emb_loss:
						num_of_concat_mention_context_span_embeds = 3
						context_funct_hidden_size = num_of_concat_mention_context_span_embeds*4
						self.context_funct_h1 = nn.Linear(config.hidden_size*num_of_concat_mention_context_span_embeds, config.hidden_size*context_funct_hidden_size)
						#self.context_funct_h2 = nn.Linear(config.hidden_size*context_funct_hidden_size, config.hidden_size*context_funct_hidden_size)
						self.context_funct_o = nn.Linear(config.hidden_size*context_funct_hidden_size, config.hidden_size)
						self.context_act_funct = nn.GELU()
						self.context_layer_norm = nn.LayerNorm(config.hidden_size*context_funct_hidden_size, eps=config.layer_norm_eps)
					
					if self.enable_predicate_span:
						num_of_concat_predicate_span_embeds = 3
						predicate_funct_hidden_size = num_of_concat_predicate_span_embeds*4
						self.predicate_funct_h1 = nn.Linear(config.hidden_size*num_of_concat_predicate_span_embeds, config.hidden_size*predicate_funct_hidden_size)
						#self.predicate_funct_h2 = nn.Linear(config.hidden_size*predicate_funct_hidden_size, config.hidden_size*predicate_funct_hidden_size)
						self.predicate_funct_o = nn.Linear(config.hidden_size*predicate_funct_hidden_size, config.hidden_size)
						self.predicate_act_funct = nn.GELU()
						self.predicate_layer_norm = nn.LayerNorm(config.hidden_size*predicate_funct_hidden_size, eps=config.layer_norm_eps)

					if self.enable_pos_emb_loss:
						num_of_concat_span_embeds = 2
						#self.span_classifier = nn.Linear(config.hidden_size*num_of_concat_span_embeds, config.num_labels)
						
						pos_emb_funct_hidden_size = num_of_concat_span_embeds*4
						self.pos_emb_funct_h1 = nn.Linear(config.hidden_size*num_of_concat_span_embeds, config.hidden_size*pos_emb_funct_hidden_size)
						#self.pos_emb_funct_h2 = nn.Linear(config.hidden_size*pos_emb_funct_hidden_size, config.hidden_size*pos_emb_funct_hidden_size)
						self.pos_emb_funct_o = nn.Linear(config.hidden_size*pos_emb_funct_hidden_size, config.hidden_size)
						self.pos_emb_act_funct = nn.GELU()
						self.pos_emb_layer_norm = nn.LayerNorm(config.hidden_size*pos_emb_funct_hidden_size, eps=config.layer_norm_eps)
						
					
				
				'''
				# representation function.
				rep_funct_hidden_size = num_of_concat_embeds*4
				self.rep_funct_h1 = nn.Linear(config.hidden_size*num_of_concat_embeds, config.hidden_size*rep_funct_hidden_size)
				self.rep_funct_h2 = nn.Linear(config.hidden_size*rep_funct_hidden_size, config.hidden_size*rep_funct_hidden_size)
				self.rep_funct_o = nn.Linear(config.hidden_size*rep_funct_hidden_size, config.hidden_size)
				
				self.rep_act_funct = nn.GELU()
				self.LayerNorm = nn.LayerNorm(config.hidden_size*rep_funct_hidden_size, eps=config.layer_norm_eps)
				self.dropout = nn.Dropout(config.hidden_dropout_prob)
				
				self.rel_classifier = nn.Linear(config.hidden_size, config.num_labels)
				
				'''

				
				
			else:
				self.rel_classifier = nn.Linear(config.hidden_size, config.num_labels)
		# [GP][START] - two classifiers are needed for joint NER-PPI task. - 04-12-2021
		elif self.finetuning_task == 'joint-ner-ppi':
			self.ner_classifier = nn.Linear(config.hidden_size, config.num_labels)
			if self.relation_representation == 'STANDARD_mention_pooling':
				self.rel_classifier = nn.Linear(config.hidden_size*2, self.num_ppi_labels)
			elif self.relation_representation == 'STANDARD_mention_pooling_plus_context':
				self.rel_classifier = nn.Linear(config.hidden_size*3, self.num_ppi_labels)
			else:
				self.rel_classifier = nn.Linear(config.hidden_size, self.num_ppi_labels)
		# [GP][END] - two classifiers are needed for joint NER-PPI task. - 04-12-2021
		else:
			self.classifier = nn.Linear(config.hidden_size, config.num_labels)

		self.init_weights()

	@add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
	@add_code_sample_docstrings(
		processor_class=_TOKENIZER_FOR_DOC,
		checkpoint=_CHECKPOINT_FOR_DOC,
		output_type=TokenClassifierOutput,
		config_class=_CONFIG_FOR_DOC,
	)
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
		
		# [GP][START] - added a EM_entity_start parameter for PPI task. 04-04-2021
		e1_e2_start=None,
		# [GP][END] - added a EM_entity_start parameter for PPI task. 04-04-2021
		
		# [GP][START] - added a mention pooling parameter for PPI task. 04-25-2021
		entity_mention=None,
		# [GP][END] - added a mention pooling parameter for PPI task. 04-25-2021
		
		ppi_relations=None,	
		
		
		relations=None,
		
		predicates=None,
		
		
		entity_types=None,
		
		
		
		
		directed=None,
		reverse=None,

	):
		r"""
		labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
			Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
			1]``.
		"""
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

		sequence_output = outputs[0]

		sequence_output = self.dropout(sequence_output)
		
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

		# [START][GP] - return labels for PPI classification in the joint learning. 10-01-2021
		ppi_labels = None
		# [END][GP] - return labels for PPI classification in the joint learning. 10-01-2021

		if self.finetuning_task == 'ner':
			logits = self.ner_classifier(sequence_output)
			loss = None
			if labels is not None:
				loss_fct = CrossEntropyLoss()
				# Only keep active parts of the loss
				if attention_mask is not None:
					active_loss = attention_mask.view(-1) == 1
					active_logits = logits.view(-1, self.num_labels)
					active_labels = torch.where(
						active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
					)
					loss = loss_fct(active_logits, active_labels)
		
					'''
					print('sequence_output.shape:', sequence_output.shape)
					print('logits.shape:', logits.shape)
					print('labels.shape:', labels.shape)
					print('active_logits.shape:', active_logits.shape)
					print('active_labels.shape:', active_labels.shape)
					input('enter...')
					'''

				else:
					loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
				
				#loss = self.ner_weight*loss


		elif self.finetuning_task == 'ppi':
			
			# loss calculation using Entity START markers. 04-04-2021
			if self.relation_representation == 'EM_entity_start': 
				# ref: https://github.com/plkmo/BERT-Relation-Extraction/blob/master/src/model/BERT/modeling_bert.py
				### two heads: LM and blanks ###
				blankv1v2 = sequence_output[:, e1_e2_start, :]

				buffer = []
				for i in range(blankv1v2.shape[0]): # iterate batch & collect
					
					v1v2 = blankv1v2[i, i, :, :]
					v1v2 = torch.cat((v1v2[0], v1v2[1]))
					buffer.append(v1v2)
						
						
					if self.training: # if it's training,
						

						# For undirected (symmetric) relations, consider both A-B and B-A. 11-05-2021
						if directed[i] == False:
							v2v1 = blankv1v2[i, i, :, :]
							v2v1 = torch.cat((v2v1[1], v2v1[0]))
							buffer.append(v2v1)
							
							first_half = labels[0:i,:]
							second_half = labels[i:,:]
							i_label = torch.unsqueeze(labels[i,:], 0)
							labels = torch.cat([first_half, i_label, second_half], 0)
					
					'''
					
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




				del blankv1v2
				v1v2 = torch.stack([a for a in buffer], dim=0)
				del buffer
	
				logits = self.rel_classifier(v1v2)
				
				
				
				
				
				

			elif self.relation_representation == 'EM_entity_start_plus_context':
				buffer = []
				for i in range(sequence_output.shape[0]): # iterate batch & collect
					
					e1_start_em_idx = e1_e2_start[i][0]
					e2_start_em_idx = e1_e2_start[i][1]
					e1_start_em = sequence_output[i, e1_start_em_idx, :]
					e2_start_em = sequence_output[i, e2_start_em_idx, :]

					e1_start = entity_mention[i][0]
					e1_end   = entity_mention[i][1]
					e2_start = entity_mention[i][2]
					e2_end   = entity_mention[i][3]

					# if entity 1 appears before entity 2 in the sentence, and there is a context between them,
					if e1_end + 1 < e2_start:
						context = sequence_output[i, e1_end:e2_start, :]
						context = torch.transpose(context, 0, 1)
						context = torch.max(context, dim=1)[0] # max_pooling
					# if entity 2 appears before entity 1 in the sentence, and there is a context between them,
					elif e2_end + 1 < e1_start:
						context = sequence_output[i, e2_end:e1_start, :]
						context = torch.transpose(context, 0, 1)
						context = torch.max(context, dim=1)[0] # max_pooling
					else:
						context = torch.zeros([self.hidden_size], dtype=sequence_output.dtype, device=sequence_output.device)

					em_plus_context = torch.cat((e1_start_em, context))
					em_plus_context = torch.cat((em_plus_context, e2_start_em))
							
					buffer.append(em_plus_context)

					'''
					for idx in input_ids:
						print('input_ids:', idx)
					for idx, elem in enumerate(input_ids):	
						print('input:', self.tokenizer.convert_ids_to_tokens(elem))
					print('input_ids.size():', input_ids.size())
					print('e1_start_em_idx:', e1_start_em_idx, '/ e2_start_em_idx:', e2_start_em_idx)
					print('e1_start:', e1_start, '/ e1_end:', e1_end, '/ e2_start:', e2_start, '/ e2_end:', e2_end)
					print('e1_start_em.shape:', e1_start_em.shape)
					print('e2_start_em.shape:', e2_start_em.shape)
					print('context.shape:', context.shape)
					print('em_plus_context.shape:', em_plus_context.shape)
					input('enter..')
					'''
					
					del e1_start_em
					del e2_start_em
					del context
					del em_plus_context
					
				em_start_plus_context = torch.stack([x for x in buffer], dim=0)
				del buffer

				#logits = self.rel_classifier(em_start_plus_context)
				
				
				
					
				z = self.LayerNorm(self.rep_act_funct(self.rep_funct_h1(em_start_plus_context)))
				#z = self.LayerNorm(self.rep_act_funct(self.rep_funct_h2(z)))
				z = self.rep_funct_o(z)
				z = self.dropout(z)

				logits = self.rel_classifier(z)
					
				
				
				
				
				
				
				
				
				

			# loss calculation using mention pooling. 04-25-2021
			# TODO: 'EM_mention_pooling' hasn't been tested.
			elif self.relation_representation in ['STANDARD_mention_pooling', 'EM_mention_pooling']:
				buffer = []
				for i in range(sequence_output.shape[0]): # iterate batch & collect
					e1_start = entity_mention[i][0]
					e1_end   = entity_mention[i][1]
					e2_start = entity_mention[i][2]
					e2_end   = entity_mention[i][3]
					
					e1_rep = sequence_output[i, e1_start:e1_end, :]
					e2_rep = sequence_output[i, e2_start:e2_end, :]

					e1_rep = torch.transpose(e1_rep, 0, 1)
					e1_rep = torch.max(e1_rep, dim=1)[0] # max_pooling
						
					e2_rep = torch.transpose(e2_rep, 0, 1)
					e2_rep = torch.max(e2_rep, dim=1)[0] # max_pooling
					
					e1_e2_rep = torch.cat((e2_rep, e2_rep))
					
					buffer.append(e1_e2_rep)

					'''
					print(e1_start, e1_end, e2_start, e2_end)
					print(e1_rep)
					print(e2_rep)
					print(e1_e2_rep)
					print(e1_e2_rep.shape)
					input('enter..')
					'''

					del e1_rep
					del e2_rep
					del e1_e2_rep
					
				mention_pooling = torch.stack([x for x in buffer], dim=0)
				del buffer

				logits = self.rel_classifier(mention_pooling)
			
			# loss calculation using mention pooling plus context. 05-07-2021
			elif self.relation_representation in ['STANDARD_mention_pooling_plus_context']:
				buffer = []
				for i in range(sequence_output.shape[0]): # iterate batch & collect
					
					
					e1_start = entity_mention[i][0]
					e1_end   = entity_mention[i][1]
					e2_start = entity_mention[i][2]
					e2_end   = entity_mention[i][3]
					
					e1_rep = sequence_output[i, e1_start:e1_end, :]
					e2_rep = sequence_output[i, e2_start:e2_end, :]
					
					'''
					# debug
					print(e1_rep.shape)
					print(e1_rep)
					print(torch.transpose(e1_rep, 0, 1))
					print(torch.transpose(e1_rep, 0, 1).shape)
					e1_rep = torch.transpose(e1_rep, 0, 1)
					print(torch.max(e1_rep, dim=1))
					print(torch.max(e1_rep, dim=1).shape)
					print(torch.max(e1_rep, dim=1)[0])
					print(torch.max(e1_rep, dim=1)[0].shape)
					print(torch.mean(e1_rep, dim=1))
					print(torch.mean(e1_rep, dim=1).shape)
					input('enter..')
					'''
					
					e1_rep = torch.transpose(e1_rep, 0, 1)
					e1_rep = torch.max(e1_rep, dim=1)[0] # max_pooling [0] -> values, [1] -> indices
					#e1_rep = torch.mean(e1_rep, dim=1) # mean_pooling

					e2_rep = torch.transpose(e2_rep, 0, 1)
					e2_rep = torch.max(e2_rep, dim=1)[0] # max_pooling [0] -> values, [1] -> indices
					#e2_rep = torch.mean(e2_rep, dim=1) # mean_pooling
					
					# if entity 1 appears before entity 2 in the sentence, and there is a context between them,
					if e1_end + 1 < e2_start:
						context = sequence_output[i, e1_end:e2_start, :]
						context = torch.transpose(context, 0, 1)
						context = torch.max(context, dim=1)[0] # max_pooling
					# if entity 2 appears before entity 1 in the sentence, and there is a context between them,
					elif e2_end + 1 < e1_start:				
						context = sequence_output[i, e2_end:e1_start, :]
						context = torch.transpose(context, 0, 1)
						context = torch.max(context, dim=1)[0] # max_pooling
					else:
						context = torch.zeros([self.hidden_size], dtype=sequence_output.dtype, device=sequence_output.device)
					
					'''
					e1_e2_plus_context = torch.cat((e1_rep, context))
					e1_e2_plus_context = torch.cat((e1_e2_plus_context, e2_rep))
					
					buffer.append(e1_e2_plus_context)

					del e1_rep
					del e2_rep
					del context
					del e1_e2_plus_context
	
				mention_pooling = torch.stack([x for x in buffer], dim=0)
				del buffer

				logits = self.rel_classifier(mention_pooling)
				'''
				
					
				
					'''
					e1_start = entity_mention[i][0]
					e1_end   = entity_mention[i][1]
					e2_start = entity_mention[i][2]
					e2_end   = entity_mention[i][3]
					
					e1_span_s_rep = sequence_output[i, e1_start-1, :]
					e1_span_e_rep = sequence_output[i, e1_end, :]
					e2_span_s_rep = sequence_output[i, e2_start-1, :]
					e2_span_e_rep = sequence_output[i, e2_end, :]
				
					e_span_rep = torch.cat((e1_span_s_rep, context))
					e_span_rep = torch.cat((e_span_rep, e2_span_s_rep))
					#e_span_rep = torch.cat((e_span_rep, e2_span_s_rep))
					#e_span_rep = torch.cat((e_span_rep, e2_span_e_rep))

					buffer.append(e_span_rep)

					del e1_span_s_rep
					del e1_span_e_rep
					del e2_span_s_rep
					del e2_span_e_rep
					del e_span_rep
					'''
					
					
					
					# using span positional embeds.
					'''
					input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i])
					
					print(input_tokens)
					
					print(e1_start)
					print(e1_end)
					print(e2_start)
					print(e2_end)
					input('enter..')
					'''
					
					#cls_token = sequence_output[i, 0, :]
					
					e1_span_s_pos_embed = self.bert.embeddings.position_embeddings(e1_start-1)
					e1_span_e_pos_embed = self.bert.embeddings.position_embeddings(e1_end)
					e2_span_s_pos_embed = self.bert.embeddings.position_embeddings(e2_start-1)
					e2_span_e_pos_embed = self.bert.embeddings.position_embeddings(e2_end)
					
					e_span_rep = torch.cat((e1_span_s_pos_embed, e1_rep))
					#e_span_rep = torch.cat((e_span_rep, e1_span_e_pos_embed))
					#e_span_rep = torch.cat((e_span_rep, context))
					e_span_rep = torch.cat((e_span_rep, e2_span_s_pos_embed))
					e_span_rep = torch.cat((e_span_rep, e2_rep))
					#e_span_rep = torch.cat((e_span_rep, e2_span_e_pos_embed))
					
					
					buffer.append(e_span_rep)
					
					del e1_rep
					del e2_rep
					del context
					del e1_span_s_pos_embed
					del e2_span_s_pos_embed
					del e_span_rep

					
					
					
					
					
				mention_pooling = torch.stack([x for x in buffer], dim=0)
				del buffer

				#logits = self.rel_classifier(mention_pooling)
					
					
					
				z = self.LayerNorm(self.rep_act_funct(self.rep_funct_h1(mention_pooling)))
				z = self.dropout(z)
				z = self.LayerNorm(self.rep_act_funct(self.rep_funct_h2(z)))
				z = self.dropout(z)
				z = self.rep_funct_o(z)
				z = self.dropout(z)

				logits = self.rel_classifier(z)
				
				
				
			
			# loss calculation using CLS token.
			# TODO: 'EM_cls_token' hasn't been tested.
			elif self.relation_representation in ['STANDARD_cls_token', 'EM_cls_token']:
				cls_token = sequence_output[:, 0, :]
				logits = self.rel_classifier(cls_token)
			
			
			
			# loss calculation using POSITIONAL_mention_pooling_plus_context token. 11-10-2021
			elif self.relation_representation in ['POSITIONAL_mention_pooling_plus_context']:

				rel_rep_buffer_list = [] 	# relation representaion buffer per each sample
				rel_label_buffer_list = [] 	# relation label buffer per each sample
				total_rel_count = 0
				
				
				#if self.enable_pos_emb_loss:
				#	rel_span_buffer_list = []
			
				
				for i in range(sequence_output.shape[0]): # iterate batch & collect
					rel_list = [x for x in torch.split(relations[i], 5) if -100 not in x] # e1_span_s, e1_span_e, e2_span_s, e2_span_e, rel['rel_id']
					#predicate_list = [x for x in torch.split(predicates[i], 2) if -100 not in x] # predicates_info
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
					#if len(rel_list) > 1:
					print('relations[i]:', relations[i])
					print('relations[i].shape:', relations[i].shape)
					print('rel_list:', rel_list)
					print('predicates[i]:', predicates[i])
					print('predicate_list:', predicate_list)
					#input('enter...')
					'''
					
					
					
					entity_type_list = [x for x in torch.split(entity_types[i], 2) if -100 not in x]
					
					
					
					# debug
					if len(rel_list) != len(predicate_list) != len(entity_type_list):
						print('relations[i]:', relations[i])
						print('relations[i].shape:', relations[i].shape)
						print('rel_list:', rel_list)
						print('predicates[i]:', predicates[i])
						print('predicate_list:', predicate_list)
						print(len(rel_list))
						print(len(predicate_list))
						print('entity_types[i]:', entity_types[i])
						print('entity_type_list:', entity_type_list)
						input('enter...')
					
					
					rel_rep_buffer = [] 	# relation representaion buffer per each sample
					rel_label_buffer = [] 	# ppi label buffer per each sample
					
					#if self.enable_pos_emb_loss:
					#	rel_span_buffer = []
					
					for rel, predicate, entity_type in zip(rel_list, predicate_list, entity_type_list):
						#rel = rel.tolist()
						
						e1_start  = rel[0]
						e1_end    = rel[1]
						e2_start  = rel[2]
						e2_end    = rel[3]
						rel_label = rel[4]
						
						e1_rep = sequence_output[i, e1_start:e1_end, :]
						e2_rep = sequence_output[i, e2_start:e2_end, :]
						
						
						# debug
						'''
						print('sequence_output:', sequence_output)
						print('e1_rep.shape:', e1_rep.shape)
						print('e1_rep:', e1_rep)
						print('torch.transpose(e1_rep, 0, 1):', torch.transpose(e1_rep, 0, 1))
						print('torch.transpose(e1_rep, 0, 1).shape:', torch.transpose(e1_rep, 0, 1).shape)
						e1_rep_transposed = torch.transpose(e1_rep, 0, 1)
						print('torch.max(e1_rep_transposed, dim=1):', torch.max(e1_rep_transposed, dim=1))
						print('type(torch.max(e1_rep_transposed, dim=1)[0]):', type(torch.max(e1_rep_transposed, dim=1)[0]))
						print('len(torch.max(e1_rep_transposed, dim=1)[0]):', len(torch.max(e1_rep_transposed, dim=1)[0]))
						original_values   = torch.max(e1_rep, dim=0)[0].tolist()
						transposed_values = torch.max(e1_rep_transposed, dim=1)[0].tolist()
						print('original_values:', original_values)
						print('transposed_values:', transposed_values)
						print('bool(original_values == transposed_values):', bool(original_values == transposed_values))
						print(torch.max(e1_rep, dim=1)[0])
						print(torch.max(e1_rep, dim=1)[0].shape)
						print(torch.mean(e1_rep, dim=1))
						print(torch.mean(e1_rep, dim=1).shape)
						input('enter..')
						'''

						e1_rep = torch.max(e1_rep, dim=0)[0] # max_pooling [0] -> values, [1] -> indices
						#e1_rep = torch.mean(e1_rep, dim=1)  # mean_pooling
						
						e2_rep = torch.max(e2_rep, dim=0)[0] # max_pooling [0] -> values, [1] -> indices
						#e2_rep = torch.mean(e2_rep, dim=1)  # mean_pooling
						
						# if entity 1 appears before entity 2 in the sentence, and there is a context between them,
						if e1_end < e2_start:
							context = sequence_output[i, e1_end:e2_start, :]
						# if entity 2 appears before entity 1 in the sentence, and there is a context between them,
						elif e2_end < e1_start:				
							context = sequence_output[i, e2_end:e1_start, :]
						else:
							context = torch.zeros(1, self.hidden_size, dtype=sequence_output.dtype, device=sequence_output.device)
						
						context = torch.max(context, dim=0)[0] # max_pooling
						
						'''
						# if entity 1 appears before entity 2 in the sentence, and there is a context between them,
						if e1_end + 1 < e2_start:
							context = sequence_output[i, e1_end:e2_start, :]
							context = torch.max(context, dim=0)[0] # max_pooling
						# if entity 2 appears before entity 1 in the sentence, and there is a context between them,
						elif e2_end + 1 < e1_start:				
							context = sequence_output[i, e2_end:e1_start, :]
							context = torch.max(context, dim=0)[0] # max_pooling
						else:
							context = torch.zeros([self.hidden_size], dtype=sequence_output.dtype, device=sequence_output.device)
						
						if self.enable_predicate_span:
							use_predicate_span = predicate[2]
							
							if use_predicate_span:
								predicate_span_s, predicate_span_e = predicate[0], predicate[1]
								
								context_2 = sequence_output[i, predicate_span_s:predicate_span_e, :]
								context_2 = torch.max(context_2, dim=0)[0] # max_pooling
							else:
								context_2 = torch.zeros([self.hidden_size], dtype=sequence_output.dtype, device=sequence_output.device)
						'''

						if self.enable_ffnn_for_rep:
							mention_context_rep = torch.cat((e1_rep, context))
							mention_context_rep = torch.cat((mention_context_rep, e2_rep))
						
							mention_context_rep_z = self.context_layer_norm(self.context_act_funct(self.context_funct_h1(mention_context_rep)))
							mention_context_rep_z = self.dropout(mention_context_rep_z)
							#mention_context_rep_z = self.context_layer_norm(self.context_act_funct(self.context_funct_h2(mention_context_rep_z)))
							#mention_context_rep_z = self.dropout(mention_context_rep_z)
							mention_context_rep_z = self.context_funct_o(mention_context_rep_z)
							mention_context_rep_z = self.dropout(mention_context_rep_z)
							

						
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

							if self.enable_ffnn_for_rep:
								mention_predicate_rep = torch.cat((e1_rep, predicate_span))
								mention_predicate_rep = torch.cat((mention_predicate_rep, e2_rep))
							
								mention_predicate_rep_z = self.predicate_layer_norm(self.predicate_act_funct(self.predicate_funct_h1(mention_predicate_rep)))
								mention_predicate_rep_z = self.dropout(mention_predicate_rep_z)
								#mention_predicate_rep_z = self.predicate_layer_norm(self.predicate_act_funct(self.predicate_funct_h2(mention_predicate_rep_z)))
								#mention_predicate_rep_z = self.dropout(mention_predicate_rep_z)
								mention_predicate_rep_z = self.predicate_funct_o(mention_predicate_rep_z)
								mention_predicate_rep_z = self.dropout(mention_predicate_rep_z)
						
						
						
						
						if self.enable_predicate_attention:
								
							use_predicate_span = predicate[0]
							
							if use_predicate_span:
								p_s_l = zip(*[iter(predicate[1:])]*2)
								all_predicate_spans = None
								for predicate_span_s, predicate_span_e in p_s_l:
									predicate_span_s = torch.tensor(predicate_span_s, dtype=torch.int, device=sequence_output.device)
									predicate_span_e = torch.tensor(predicate_span_e, dtype=torch.int, device=sequence_output.device)
									
									predicate_span = sequence_output[i, predicate_span_s:predicate_span_e, :]
									
									'''
									#print(context)
									print('predicate_span:', predicate_span)
									print('predicate_span.shape:', predicate_span.shape)
									print(torch.cat((predicate_span, predicate_span), dim=1))
									print(torch.cat((predicate_span, predicate_span), dim=1).shape)
									#print(context.shape)
									#print(predicate_span.shape)
									#print(e1_start, e1_end, e2_start, e2_end)
									#print(predicate_span_s, predicate_span_e)
									input('enter..')
									'''
									
									
									predicate_span = torch.cat((predicate_span, predicate_span), dim=1)
									
									
									if all_predicate_spans is None:
										all_predicate_spans = predicate_span
									else:
										all_predicate_spans = torch.cat((all_predicate_spans, predicate_span))
									
									del predicate_span

								#predicate_span = torch.max(all_predicate_spans, dim=0)[0] # max_pooling
								#del all_predicate_spans
								
								
								
								concat_e1_e2 = torch.cat((e1_rep, e2_rep))
								
								#concat_e1_e2_z = self.ent_concat_ffnn(concat_e1_e2) # worse than the code below
								'''
								concat_e1_e2_z = self.context_layer_norm(self.context_act_funct(self.context_funct_h1(concat_e1_e2)))
								concat_e1_e2_z = self.dropout(concat_e1_e2_z)
								concat_e1_e2_z = self.context_layer_norm(self.context_act_funct(self.context_funct_h2(concat_e1_e2_z)))
								concat_e1_e2_z = self.dropout(concat_e1_e2_z)
								concat_e1_e2_z = self.context_funct_o(concat_e1_e2_z)
								concat_e1_e2_z = self.dropout(concat_e1_e2_z)
								'''

								
								'''
								#print('e1_rep.shape:', e1_rep.shape)
								#print('concat_e1_e2.shape:', concat_e1_e2.shape)
								#print('all_predicate_spans.shape:', all_predicate_spans.shape)
								import torch.nn.functional as F
								padded_size = concat_e1_e2.size(dim=0) - all_predicate_spans.size(dim=1)
								#print('padded_size:', padded_size)
								all_predicate_spans_result = F.pad(input=all_predicate_spans, pad=(0, padded_size), mode='constant', value=0)
								#print('all_predicate_spans_result.shape:', all_predicate_spans_result.shape)
								#print('all_predicate_spans_result:', all_predicate_spans_result)
								'''
								
								predicate_outputs = self.predicate_attention(
														
														concat_e1_e2=torch.unsqueeze(concat_e1_e2, 0),
														predicates=all_predicate_spans
														
													)
													
													
								
								#print(predicate_outputs)
								#print(predicate_outputs[0].shape)
								
								predicate_outputs = predicate_outputs[0].squeeze()
								#predicate_outputs = predicate_outputs[1000:]	
								
								#print(predicate_outputs)
								#print(predicate_outputs.shape)
								#input('enter...')
								
							else:
								#predicate_outputs = torch.zeros([self.hidden_size], dtype=sequence_output.dtype, device=sequence_output.device)
								predicate_outputs = torch.zeros([self.hidden_size*2], dtype=sequence_output.dtype, device=sequence_output.device)
								
								
								#import torch.nn.functional as F
								#predicate_outputs = F.pad(input=context, pad=(0, self.hidden_size), mode='constant', value=0)
								
								
								#predicate_outputs = context



						
						
						
						# using span positional embeds.
						'''
						input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i])
						
						print(input_tokens)
						
						print(e1_start)
						print(e1_end)
						print(e2_start)
						print(e2_end)
						input('enter..')
						'''
						
						#cls_token = sequence_output[i, 0, :]
						
						
						'''
						#e_span_rep = torch.cat((e1_rep, e2_rep))
						#e_span_rep = torch.cat((e1_span_s_pos_embed, e1_rep))
						#e_span_rep = torch.cat((e_span_rep, e1_span_e_pos_embed))
						e_span_rep = torch.cat((e1_rep, context))
						
						
						#if self.enable_predicate_span:
						#	e_span_rep = torch.cat((e_span_rep, context_2))
						
						
						#e_span_rep = torch.cat((e_span_rep, e2_span_s_pos_embed))
						e_span_rep = torch.cat((e_span_rep, e2_rep))
						#e_span_rep = torch.cat((e_span_rep, e2_span_e_pos_embed))
						'''
						
						if self.enable_pos_emb_loss:
							e1_span_s_pos_embed = self.bert.embeddings.position_embeddings(e1_start-1)
							e1_span_e_pos_embed = self.bert.embeddings.position_embeddings(e1_end)
							e2_span_s_pos_embed = self.bert.embeddings.position_embeddings(e2_start-1)
							e2_span_e_pos_embed = self.bert.embeddings.position_embeddings(e2_end)
							
							pos_emb_rep = torch.cat((e1_span_s_pos_embed, e2_span_s_pos_embed))
							#pos_emb_rep = torch.cat((pos_emb_rep, e2_span_s_pos_embed))
							#pos_emb_rep = torch.cat((pos_emb_rep, e2_span_e_pos_embed))
							
							
							
							
							
							
							#pos_diff_embed = self.pos_diff_embeddings(abs(e2_start - e1_start))

							pos_diff_embed = self.pos_diff_embeddings(abs(torch.round((e1_start + e1_end)/2 - (e2_start + e2_end)/2).long()))
							
							
							pos_emb_rep = torch.cat((pos_emb_rep, pos_diff_embed))
							

							
							
							
							
							if self.enable_ffnn_for_rep:
								pos_emb_rep_z = self.pos_emb_layer_norm(self.pos_emb_act_funct(self.pos_emb_funct_h1(pos_emb_rep)))
								pos_emb_rep_z = self.dropout(pos_emb_rep_z)
								#pos_emb_rep_z = self.pos_emb_layer_norm(self.pos_emb_act_funct(self.pos_emb_funct_h2(pos_emb_rep_z)))
								#pos_emb_rep_z = self.dropout(pos_emb_rep_z)
								pos_emb_rep_z = self.pos_emb_funct_o(pos_emb_rep_z)
								pos_emb_rep_z = self.dropout(pos_emb_rep_z)
								
								#e_span_rep = torch.cat((e_span_rep, z))
								
						
						
						
						if self.enable_entity_type_emb:
							e1_type_id = entity_type[0]
							e2_type_id = entity_type[1]
							
							
							e1_type_start_emb = self.entity_type_embeddings(e1_type_id)
							e1_type_end_emb = self.entity_type_embeddings(self.num_entity_types + e1_type_id)
							e2_type_start_emb = self.entity_type_embeddings(e2_type_id)
							e2_type_end_emb = self.entity_type_embeddings(self.num_entity_types + e2_type_id)
							
							#print('self.num_entity_types:', self.num_entity_types)
							
							e1_span_s_pos_embed = self.bert.embeddings.position_embeddings(e1_start-1)
							e1_span_e_pos_embed = self.bert.embeddings.position_embeddings(e1_end)
							e2_span_s_pos_embed = self.bert.embeddings.position_embeddings(e2_start-1)
							e2_span_e_pos_embed = self.bert.embeddings.position_embeddings(e2_end)
							
							e1_type_start_emb += e1_span_s_pos_embed
							e1_type_end_emb += e1_span_e_pos_embed
							e2_type_start_emb += e2_span_s_pos_embed
							e2_type_end_emb += e2_span_e_pos_embed
							
							
							
							
							
							
							

						
						
						'''
						print(e1_rep.shape)
						print(e_span_rep.shape)
						print(labels.shape)
						input('enter..')
						'''
						if self.enable_ffnn_for_rep:
							if self.enable_predicate_span and self.enable_pos_emb_loss:
								final_rep = torch.cat((mention_context_rep_z, mention_predicate_rep_z))
								final_rep = torch.cat((final_rep, pos_emb_rep_z))
								#final_rep = torch.cat((e1_rep, context))
								#final_rep = torch.cat((final_rep, e2_rep))
								#final_rep = torch.cat((final_rep, mention_predicate_rep_z))
								#final_rep = torch.cat((final_rep, pos_emb_rep_z))
							elif self.enable_predicate_span:
								final_rep = torch.cat((mention_context_rep_z, mention_predicate_rep_z))
							elif self.enable_pos_emb_loss:
								final_rep = torch.cat((mention_context_rep_z, pos_emb_rep_z))
							else:
								final_rep = torch.cat((e1_rep, context))
								final_rep = torch.cat((final_rep, e2_rep))
						else:
							
							if self.enable_predicate_span:
								context = torch.stack((context, predicate_span))
								context = torch.max(context, dim=0)[0] # max_pooling
							
							
							
							if self.enable_predicate_attention:
								
								final_rep = torch.cat((e1_rep, context))
								final_rep = torch.cat((final_rep, e2_rep))
								final_rep = torch.cat((final_rep, predicate_outputs))
								'''
								final_rep = torch.cat((e1_rep, predicate_outputs))
								final_rep = torch.cat((final_rep, e2_rep))
								'''
							elif self.enable_entity_type_emb:

								final_rep = torch.cat((e1_type_start_emb, e1_rep))
								final_rep = torch.cat((final_rep, e1_type_end_emb))
								final_rep = torch.cat((final_rep, context))
								final_rep = torch.cat((final_rep, e2_type_start_emb))
								final_rep = torch.cat((final_rep, e2_rep))
								final_rep = torch.cat((final_rep, e2_type_end_emb))
								
							else:
								final_rep = torch.cat((e1_rep, context))
								final_rep = torch.cat((final_rep, e2_rep))
								
							
							
							'''
							e1_span_width_embed = self.span_width_embeddings(abs(e1_end - e1_start))
							e2_span_width_embed = self.span_width_embeddings(abs(e2_end - e2_start))
							
							final_rep = torch.cat((e1_rep, e1_span_width_embed))
							final_rep = torch.cat((final_rep, context))
							final_rep = torch.cat((final_rep, e2_rep))
							final_rep = torch.cat((final_rep, e2_span_width_embed))
							'''



							if self.enable_pos_emb_loss:
								#final_rep = torch.cat((final_rep, pos_emb_rep))
								final_rep = torch.cat((final_rep, pos_diff_embed))
							
							
							
								
							
						'''
						print(e1_rep.shape)
						print(context.shape)
						print(e2_rep.shape)
						print(pos_emb_rep_z.shape)
						print(final_rep.shape)
						print(labels.shape)
						input('enter..')
						'''
						
						
						
						
						rel_rep_buffer.append(final_rep)
						rel_label_buffer.append(rel_label)
						
						
						#if self.enable_pos_emb_loss:
						#	rel_span_buffer.append(e_span_rep_2)

						
						

						del e1_rep
						del e2_rep
						del context
						if self.enable_predicate_span:
							del predicate_span
						if self.enable_pos_emb_loss:
							del e1_span_s_pos_embed
							del e2_span_s_pos_embed
						#del e_span_rep
						
						#if self.enable_pos_emb_loss:
						#	del e_span_rep_2

					rel_rep_buffer_list.append(rel_rep_buffer)
					rel_label_buffer_list.append(rel_label_buffer)
					
					
					#if self.enable_pos_emb_loss:
					#	rel_span_buffer_list.append(rel_span_buffer)
					
					
					
					total_rel_count += len(rel_rep_buffer)
					
					if len(rel_rep_buffer) != len(rel_label_buffer): # debug
						raise ValueError(
							"The representation size (%d) and the label size (%d) are not the same." % (config.hidden_size, config.num_attention_heads)
						)
				
				
				
				# the ppi_relations are padded in datacollator, so the size should be the same as the other samples running on diffenent gpus.
				#max_length = len([x for x in torch.split(ppi_relations[0], 5)])
				
				
				'''
				for r_l in rel_rep_buffer_list:
					print(len(r_l))
				print('len(max(rel_rep_buffer_list, key=len)):', len(max(rel_rep_buffer_list, key=len)))
				print('max_length:', max_length)
				input('enter...')
				'''
				
				#max_length = len(max(rel_rep_buffer_list, key=len))
				#max_length = 3000
				max_length = labels.size(dim=1)			
				pad_token = None
				for i in rel_rep_buffer_list:
					for j in i:
						pad_token = torch.zeros_like(j)
						break
					if pad_token is not None:
						break
				rel_rep_buffer_list_padded = [x + [pad_token]*(max_length - len(x)) for x in rel_rep_buffer_list]
				rel_label_buffer_list_padded = [x + [-100]*(max_length - len(x)) for x in rel_label_buffer_list]

				'''
				if self.enable_pos_emb_loss:
					pad_token_2 = None
					for i in rel_span_buffer_list:
						for j in i:
							pad_token_2 = torch.zeros_like(j)
							break
						if pad_token_2 is not None:
							break
							
					rel_span_buffer_list_padded = [x + [pad_token_2]*(max_length - len(x)) for x in rel_span_buffer_list]
				'''
				
				
				'''
				print(rel_rep_buffer_list)
				print(rel_label_buffer_list)
				print(rel_rep_buffer_list_padded)
				print(rel_label_buffer_list_padded)
				'''

				#rel_rep = torch.stack([a for a in rel_rep_buffer], dim=0)

				
				# TODO: add paddings when 'i' is empty. 
				# -> No, it's not necessary. It shouldn't be empty, and if the protein entity combination is not found by NER, then assign a garbage tensor like a zero tensor.
				#    Actually, it's already padded above.
				l = []
				for i in rel_rep_buffer_list_padded:
					l.append(torch.stack([x for x in i], dim=0))
				
				rel_rep = torch.stack([x for x in l], dim=0)
				
				del rel_rep_buffer_list
				del rel_rep_buffer_list_padded
				
				'''
				if self.enable_pos_emb_loss:
					l_2 = []
					for i in rel_span_buffer_list_padded:
						l_2.append(torch.stack([x for x in i], dim=0))
					
					rel_span = torch.stack([x for x in l_2], dim=0)
					
					
					del rel_span_buffer_list
					del rel_span_buffer_list_padded
				'''
				
				
				'''
				z = self.LayerNorm(self.rep_act_funct(self.rep_funct_h1(rel_rep)))
				z = self.dropout(z)
				z = self.LayerNorm(self.rep_act_funct(self.rep_funct_h2(z)))
				z = self.dropout(z)
				z = self.rep_funct_o(z)
				z = self.dropout(z)
				'''
				
				
				ppi_labels = torch.tensor(rel_label_buffer_list_padded, dtype=torch.long, device=relations.device)
				del rel_label_buffer_list_padded

				rel_loss_fct = CrossEntropyLoss()
				
				if self.enable_mention_pooling_and_context:
					
					logits = self.rel_classifier(rel_rep)

					active_loss = ppi_labels.view(-1) != -100
					active_logits = logits.view(-1, self.num_ppi_labels)
					active_labels = torch.where(
						active_loss, ppi_labels.view(-1), torch.tensor(rel_loss_fct.ignore_index).type_as(ppi_labels)
					)

					mention_and_context_loss = rel_loss_fct(active_logits, active_labels)
					

					''' Test code for one class classification for ADE data. Not working... 11-12-2021
					print(loss)
					print(loss.shape)
					input('enter..')
					
					#print(rel_rep)
					#print(rel_rep.shape)
					#print(logits.shape)
					print('active_logits:', active_logits)
					print('active_logits.shape:', active_logits.shape)
					print('active_logits.squeeze():', active_logits.squeeze())
					print('active_logits.squeeze().shape:', active_logits.squeeze().shape)
					print('ppi_labels:', ppi_labels)
					print('ppi_labels.shape:', ppi_labels.shape)
					#print(active_labels)
					#print(active_labels.shape)
					print('active_loss:', active_loss)
					print('active_loss.shape:', active_loss.shape)
					
					b = active_loss == True
					indices = b.nonzero()
					#print(b)
					#print(indices.view(-1))
					
					
					#active_logits = active_logits.squeeze()
					print('active_logits:', active_logits)
					active_logits = torch.index_select(active_logits, 0, indices.view(-1))
					print('active_logits:', active_logits)
					print('active_logits.shape:', active_logits.shape)
					
					active_labels = torch.index_select(ppi_labels.view(-1), 0, indices.view(-1))
					#active_labels = ppi_labels.view(-1)
					print('active_labels:', active_labels)
					print('active_labels.shape:', active_labels.shape)
		
					
					active_logits = active_logits.float()
					active_labels = active_labels.float()
					
					rel_loss_fct = BCEWithLogitsLoss()
					loss = rel_loss_fct(active_logits, active_labels)
					
					print(loss)
					print(loss.shape)
					input('enter..')
					'''

				"""
				if self.enable_pos_emb_loss:
					span_logits = self.span_classifier(rel_span)
					
					active_loss = ppi_labels.view(-1) != -100
					active_logits = span_logits.view(-1, self.num_ppi_labels)
					active_labels = torch.where(
						active_loss, ppi_labels.view(-1), torch.tensor(rel_loss_fct.ignore_index).type_as(ppi_labels)
					)
					
					span_loss = rel_loss_fct(active_logits, active_labels)
					
					
					''' Test code for one class classification for ADE data. Not working... 11-12-2021
					b = active_loss == True
					indices = b.nonzero()
					
					active_logits = active_logits.squeeze()
					active_logits = torch.index_select(active_logits, 0, indices.view(-1))
					active_labels = torch.index_select(ppi_labels.view(-1), 0, indices.view(-1))
		
					active_logits = active_logits.float()
					active_labels = active_labels.float()
					
					rel_loss_fct = BCEWithLogitsLoss()
					span_loss = rel_loss_fct(active_logits, active_labels)
					'''
				"""	
				if self.enable_mention_pooling_and_context:
					loss = mention_and_context_loss
				"""
				if self.enable_pos_emb_loss:
					if self.enable_mention_pooling_and_context:
						loss += span_loss
					else:
						loss = span_loss
						logits = span_logits
				"""		
			
			'''
			elif self.relation_representation == 'Multiple_Relations':
				rel_rep_buffer = []
				label_buffer = []
				for i in range(sequence_output.shape[0]): # iterate batch & collect
					ppi_list = [x for x in torch.split(ppi_relations[i], 5) if -1 not in x]
					
					for ppi in ppi_list:
						p_l = ppi.tolist()
						
						e1_start = p_l[0]
						e1_end   = p_l[1]
						e2_start = p_l[2]
						e2_end   = p_l[3]
						
						e1_rep = sequence_output[i, e1_start:e1_end, :]
						e2_rep = sequence_output[i, e2_start:e2_end, :]
						
						e1_rep_e2_rep = torch.cat((e1_rep, e2_rep))
						e1_rep_e2_rep = torch.transpose(e1_rep_e2_rep, 0, 1)
						max_pooling = torch.max(e1_rep_e2_rep, dim=1)[0]
						
						rel_rep_buffer.append(max_pooling)

						del e1_rep
						del e2_rep
						del e1_rep_e2_rep
						del max_pooling
						
						label_buffer.append(p_l[4])
				
				rel_rep = torch.stack([a for a in rel_rep_buffer], dim=0)
				del rel_rep_buffer

				ppi_label = torch.tensor(label_buffer, dtype=torch.long, device=ppi_relations.device)
				del label_buffer

				print('rel_rep.shape:', rel_rep.shape)
				print('ppi_label.shape:', ppi_label.shape)
				input('enter..')

				logits = self.rel_classifier(rel_rep)
			'''	
			
			
			
			
			# TODO: make it cleaner later.
			if not self.relation_representation in ['POSITIONAL_mention_pooling_plus_context']:

				loss_fct = CrossEntropyLoss()
				
				#if self.relation_representation == 'Multiple_Relations':
				#	loss = loss_fct(logits, ppi_label)
				#else:
				loss = loss_fct(logits, labels.squeeze(1)[:,0]) # without paddings
				
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
		# [GP][END] - loss calculation for PPI task. 04-04-2021
		
		elif self.finetuning_task == 'joint-ner-ppi':

			# calculate NER loss
			entity_logits = self.ner_classifier(sequence_output)

			entity_loss = None
			if labels is not None:
				entity_loss_fct = CrossEntropyLoss()
				# Only keep active parts of the loss
				if attention_mask is not None:
					active_loss = attention_mask.view(-1) == 1
					active_logits = entity_logits.view(-1, self.num_labels)
					active_labels = torch.where(
						active_loss, labels.view(-1), torch.tensor(entity_loss_fct.ignore_index).type_as(labels)
					)
					entity_loss = entity_loss_fct(active_logits, active_labels)
					
										
										
							
					# debug
					'''
					print(attention_mask.shape)
					print(sequence_output.shape)
					print(entity_logits.shape)
					print(active_loss.shape)
					print(active_logits.shape)
					print(labels.shape)
					print(active_labels.shape)
					print(entity_loss.shape)
					print(torch.tensor(entity_loss_fct.ignore_index).type_as(labels))
					input('enter...')
					'''
					
					
					
				else:
					entity_loss = entity_loss_fct(entity_logits.view(-1, self.num_labels), labels.view(-1))

			
			'''
			print(sequence_output.shape)
			print(entity_logits.shape)
			print(ppi_relations.shape)
			print(ppi_relations[:,:7])
			input('enter..')
			'''
			
			
			# calculate PPI loss
			predictions = torch.argmax(entity_logits, dim=2)
			
			'''
			# Remove ignored index (special tokens)
			true_predictions = [
				[p.item() for (p, l) in zip(prediction, label) if l != -100]
				for prediction, label in zip(predictions, labels)
			]
			
			for idx, elem in enumerate(input_ids):	
				print('input:', self.tokenizer.convert_ids_to_tokens(elem))

			print('active_loss.shape:', active_loss.shape)
			print('active_logits.shape:', active_logits.shape)
			print('active_labels.shape:', active_labels.shape)
			
			print('active_loss[0]:', active_loss[0])
			print('active_logits[0]:', active_logits[0])
			print('active_labels[0]:', active_labels[0])
			print('predictions:', predictions)
			print('true_predictions:', true_predictions)
			input('enter..')
			'''
			
			rel_rep_buffer_list = [] 	# relation representaion buffer per each sample
			rel_label_buffer_list = [] 	# ppi label buffer per each sample
			total_rel_count = 0
			
			for i in range(sequence_output.shape[0]): # iterate batch & collect
				input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i])
				pred = predictions[i]
				ppi_list = [x for x in torch.split(ppi_relations[i], 5) if -100 not in x]

				# debug - TODO: delete this!! this causes a heavy load.
				'''
				for p in torch.split(ppi_relations[i], 5):
					if -1 in p:
						for pp in p:
							if pp != -1: # ppi set must be the length of 5.
								print('ppi error!!!: ', pp)
								input('enter..')
				'''
				if labels is not None:
					true = labels[i]
					
					# debug - TODO: delete this!! this causes a heavy load.
					'''
					b_prot = [x for x in labels[i] if x == 0]
					if len(b_prot) < 2: # there must be at least 2 B-PROT in true label.
						print('b_prot error!!!: ', b_prot)
						input('enter..')
					'''
					match = pred == true
					match = match.nonzero().view(-1)
					'''
					protein_indice = []
					for j in match: # j == pred, true, input_tokens index
						if pred[j] == 0: # 0: B-PROT
							protein = input_tokens[j] # debug
							for k in range(j+1, len(pred)):
								next_token = input_tokens[k]
								if next_token.startswith('##') or pred[k] == 1: # 1: I-PROT
									protein += next_token # debug
								else:
									break
							#print('protein:', protein) # debug
							protein_indice.append((j.item(), k))
					'''
				'''
				0 B-PROT
				1 I-PROT
				2 O
				'''
				pred_protein_indice = []
				true_protein_indice = []
				for j, l in enumerate(pred):
					if input_tokens[j] in self.tokenizer.all_special_tokens: # skip special tokens.
						continue
					
					if l == 0: # 0: B-PROT
						#protein = self.tokenizer.convert_ids_to_tokens(input_ids[i][j].item()) # debug
						first_token = input_tokens[j]
						if first_token.startswith('##'):
							continue
							
						protein = first_token # debug
						for k in range(j+1, len(pred)):
							if input_tokens[k] in self.tokenizer.all_special_tokens: # skip special tokens.
								break
							#next_token = self.tokenizer.convert_ids_to_tokens(input_ids[i][k].item())
							next_token = input_tokens[k]
							if next_token.startswith('##') or pred[k] == 1: # 1: I-PROT
								protein += next_token # debug
							else:
								break
						
						has_alphabet = False
						for m in input_tokens[j:k]: # check if any of the tokens has an alphabet.
							if bool(re.search("[a-zA-Z]", m)):
								has_alphabet = True
								break
						
						if has_alphabet:
							#print('protein:', protein) # debug
							pred_protein_indice.append((j, k))
							
							if labels is not None:
								if j in match:
									true_protein_indice.append((j, k))
				# debug
				'''
				print('>> self.tokenizer.all_special_tokens:', self.tokenizer.all_special_tokens)
				#print('len(input_tokens):', len(input_tokens), '/ len(pred):', len(pred), '/ len(true):', len(true))
				print('>> len(input_ids):', len(input_ids[i]), '/ len(pred):', len(pred), '/ len(true):', len(labels[i]))
				print('>> pred tokens:')
				print(pred)
				print('>> true tokens:')
				print(true)
				print('>> match:', match)
				print('>> input_tokens:')
				print(input_tokens)
				for ttt in pred_protein_indice:
					print('>> pred protein - idx:', ttt, '/ token:', input_tokens[ttt[0]:ttt[1]])
				for ttt in true_protein_indice:
					print('>> true protein - idx:', ttt, '/ token:', input_tokens[ttt[0]:ttt[1]])					
				print('>> ppi:', ppi_list)
				for ttt in ppi_list:
					print('>> ppi - e1 idx:', ttt[0], ttt[1], '/ token:', input_tokens[ttt[0].item():ttt[1].item()])
					print('>> ppi - e2 idx:', ttt[2], ttt[3], '/ token:', input_tokens[ttt[2].item():ttt[3].item()])
					print('>> ppi - relation:', ttt[4])
				#print('ppi_relations:', ppi_relations[i])
				#input('enter..')
				'''
				
				
				
				
				
				rel_rep_buffer = [] 	# relation representaion buffer per each sample
				rel_label_buffer = [] 	# ppi label buffer per each sample
				
				
				
						
		
		
		
				#print('>> self.training:', self.training)
				
				
				
				
				'''
				## First, add the labeled PPIs to buffers.
				if self.relation_representation == 'STANDARD_mention_pooling':
					for ppi in ppi_list:
						p_l = ppi.tolist()
						
						e1_start = p_l[0]
						e1_end   = p_l[1]
						e2_start = p_l[2]
						e2_end   = p_l[3]
						
						e1_rep = sequence_output[i, e1_start:e1_end, :]
						e2_rep = sequence_output[i, e2_start:e2_end, :]
						
						e1_rep = torch.transpose(e1_rep, 0, 1)
						e1_rep = torch.max(e1_rep, dim=1)[0] # max_pooling
							
						e2_rep = torch.transpose(e2_rep, 0, 1)
						e2_rep = torch.max(e2_rep, dim=1)[0] # max_pooling
						
						e1_e2_rep = torch.cat((e2_rep, e2_rep))
						
						rel_rep_buffer.append(e1_e2_rep)
						rel_label_buffer.append(p_l[4])

						del e1_rep
						del e2_rep
						del e1_e2_rep
				
				elif self.relation_representation == 'STANDARD_mention_pooling_plus_context': # add a context vector.
					for ppi in ppi_list:
						p_l = ppi.tolist()
						
						e1_start = p_l[0]
						e1_end   = p_l[1]
						e2_start = p_l[2]
						e2_end   = p_l[3]
						
						e1_rep = sequence_output[i, e1_start:e1_end, :]
						e2_rep = sequence_output[i, e2_start:e2_end, :]
						
						e1_rep = torch.transpose(e1_rep, 0, 1)
						e1_rep = torch.max(e1_rep, dim=1)[0] # max_pooling
							
						e2_rep = torch.transpose(e2_rep, 0, 1)
						e2_rep = torch.max(e2_rep, dim=1)[0] # max_pooling
						
						# if entity 1 appears before entity 2 in the sentence, and there is a context between them,
						if e1_end + 1 < e2_start:
							context = sequence_output[i, e1_end:e2_start, :]
							context = torch.transpose(context, 0, 1)
							context = torch.max(context, dim=1)[0] # max_pooling
						# if entity 2 appears before entity 1 in the sentence, and there is a context between them,
						elif e2_end + 1 < e1_start:				
							context = sequence_output[i, e2_end:e1_start, :]
							context = torch.transpose(context, 0, 1)
							context = torch.max(context, dim=1)[0] # max_pooling
						else:
							context = torch.zeros([self.hidden_size], dtype=sequence_output.dtype, device=sequence_output.device)
							#print('e1_start:', e1_start, '/ e1_end:', e1_end, '/ e2_start:', e2_start, '/ e2_end:', e2_end)
							#print('ZERO tensor!!! - context.device:', context.device)
							#print('ZERO tensor!!! - context.dtype:', context.dtype)
							#input('enter..')
						
						#print('sequence_output.device:', sequence_output.device)
						#print('e1_rep.device:', e1_rep.device)
						#print('e2_rep.device:', e2_rep.device)
						#print('context.device:', context.device)
						#print('sequence_output.dtype:', sequence_output.dtype)
						#print('e1_rep.dtype:', e1_rep.dtype)
						#print('e2_rep.dtype:', e2_rep.dtype)
						#print('context.dtype:', context.dtype)
						
						e1_e2_plus_context = torch.cat((e1_rep, context))
						e1_e2_plus_context = torch.cat((e1_e2_plus_context, e2_rep))

						rel_rep_buffer.append(e1_e2_plus_context)
						rel_label_buffer.append(p_l[4])

						del e1_rep
						del e2_rep
						del context
						del e1_e2_plus_context
				'''	
					
				
				
				
				## Then, find a combination of predicted proteins by NER,
				## and add wrong relations (protein - non_protein (e.g., 'of')) to buffers as negatives.
				if self.training: # if it's training,
					for c in itertools.combinations(pred_protein_indice, 2):
					#for c in itertools.combinations(true_protein_indice, 2):
						c_flatten = [item for sublist in c for item in sublist]
						
						e1_start = c_flatten[0]
						e1_end   = c_flatten[1]
						e2_start = c_flatten[2]
						e2_end   = c_flatten[3]
						
						known_ppi = False
						are_entities_overlapped = False # check if the two predicted proteins overlap the proteins in PPI.
						is_wrong_relation = False # check if one of the two predicted proteins has a combination with a non-protein.
						ppi_label = None
						
						for ppi in ppi_list:
							p_l = ppi.tolist()
							
							'''
							print(type(c_flatten))
							print(type(p_l[:4]))
							print(c_flatten)
							print(p_l[:4])
							print(bool(c_flatten == p_l[:4]))
							input('enter..')
							'''
							
							
							# if you added labeled PPIs (ground truth) above, don't use this or skip it since it's a duplicate.
							# add a combination of predicted proteins that are the same as the entities in known (labeled) PPIs.
							if c_flatten == p_l[:4]:
								
								'''
								#print('len(input_tokens):', len(input_tokens), 'len(pred):', len(pred), 'len(true):', len(true))
								print('len(input_ids):', len(input_ids[i]), 'len(pred):', len(pred), 'len(true):', len(labels[i]))
								print('pred:', pred)
								print('true:', true)
								print('match:', match)
								print('input_tokens:', input_tokens)
								for i in protein_indice:
									print('protein - idx:', i, '/ token:', input_tokens[i[0]:i[1]])
								print('ppi_list:', ppi_list)
								for i in ppi_list:
									print('ppi - e1 idx:', i[0], i[1], '/ token:', input_tokens[i[0].item():i[1].item()])
									print('ppi - e1 idx:', i[2], i[3], '/ token:', input_tokens[i[2].item():i[3].item()])
									print('ppi - relation:', i[4])
								input('enter..')
								'''
								ppi_label = p_l[4]
								known_ppi = True
								break
							
							
							# ref: https://stackoverflow.com/questions/40367461/intersection-of-two-lists-of-ranges-in-python/40368603
							def intersections(a,b):
								ranges = []
								i = j = 0
								while i < len(a) and j < len(b):
									a_left, a_right = a[i]
									b_left, b_right = b[j]

									if a_right < b_right:
										i += 1
									else:
										j += 1

									if a_right >= b_left and b_right >= a_left:
										end_pts = sorted([a_left, a_right, b_left, b_right])
										middle = [end_pts[1], end_pts[2]]
										if end_pts[1] != end_pts[2]:
											ranges.append(middle)

								ri = 0
								while ri < len(ranges)-1:
									if ranges[ri][1] == ranges[ri+1][0]:
										ranges[ri:ri+2] = [[ranges[ri][0], ranges[ri+1][1]]]

									ri += 1

								return ranges
							

							pred_prot_indice = [[e1_start, e1_end], [e2_start, e2_end]]
							ppi_e_1_idx = [[p_l[0], p_l[1]]]
							ppi_e_2_idx = [[p_l[2], p_l[3]]]
							
							'''
							print('pred_prot_indice:', pred_prot_indice)
							print('ppi_e_1_idx:', ppi_e_1_idx)
							print(intersections(pred_prot_indice, ppi_e_1_idx))
							
							print('pred_prot_indice:', pred_prot_indice)
							print('ppi_e_2_idx:', ppi_e_2_idx)
							print(intersections(pred_prot_indice, ppi_e_2_idx))
							input('enter..')
							'''
							
							#if len(intersections(pred_prot_indice, ppi_e_1_idx)) > 0 and len(intersections(pred_prot_indice, ppi_e_2_idx)) > 0:
							#	are_entities_overlapped = True
							#	break
							
							#if ((ppi_e_1_idx[0] in pred_prot_indice) and len(intersections(pred_prot_indice, ppi_e_2_idx)) == 0) or \
							#   ((ppi_e_2_idx[0] in pred_prot_indice) and len(intersections(pred_prot_indice, ppi_e_1_idx)) == 0):
							#	is_wrong_relation = True
							#	break
							
						#if are_entities_overlapped: # skip the case where both entities overlap the proteins in the PPI.
						#	continue
			
						#if not is_wrong_relation:
						#	continue
								
						
						
						if known_ppi:
							#	continue
							#else:
								#ppi_label = 2 # if not known ppi, then regard it as negative relation. in case of (enzyme, structural, negative classes)
							#	ppi_label = 1 # if not known ppi, then regard it as negative relation. in case of (positive, negative classes)
							

							

							if self.relation_representation == 'STANDARD_mention_pooling':
								e1_rep = sequence_output[i, e1_start:e1_end, :]
								e2_rep = sequence_output[i, e2_start:e2_end, :]
								
								e1_rep_e2_rep = torch.cat((e1_rep, e2_rep))
								e1_rep_e2_rep = torch.transpose(e1_rep_e2_rep, 0, 1)
								max_pooling = torch.max(e1_rep_e2_rep, dim=1)[0]
								
								'''
								print('c_flatten.requires_grad:', c_flatten.requires_grad)
								print('e1_rep.requires_grad:', e1_rep.requires_grad)
								print('e1_rep_e2_rep.requires_grad:', e1_rep_e2_rep.requires_grad)
								print('max_pooling.requires_grad:', max_pooling.requires_grad)
								print('pred.requires_grad:', pred.requires_grad)
								'''
								
								rel_rep_buffer.append(max_pooling)
								rel_label_buffer.append(ppi_label)
							
								del e1_rep
								del e2_rep
								del e1_rep_e2_rep
								del max_pooling

							elif self.relation_representation == 'STANDARD_mention_pooling_plus_context': # add a context vector.
								e1_rep = sequence_output[i, e1_start:e1_end, :]
								e2_rep = sequence_output[i, e2_start:e2_end, :]
								
								e1_rep = torch.transpose(e1_rep, 0, 1)
								e1_rep = torch.max(e1_rep, dim=1)[0] # max_pooling
									
								e2_rep = torch.transpose(e2_rep, 0, 1)
								e2_rep = torch.max(e2_rep, dim=1)[0] # max_pooling
								
								# if entity 1 appears before entity 2 in the sentence, and there is a context between them,
								if e1_end + 1 < e2_start:
									context = sequence_output[i, e1_end:e2_start, :]
									context = torch.transpose(context, 0, 1)
									context = torch.max(context, dim=1)[0] # max_pooling
								# if entity 2 appears before entity 1 in the sentence, and there is a context between them,
								elif e2_end + 1 < e1_start:				
									context = sequence_output[i, e2_end:e1_start, :]
									context = torch.transpose(context, 0, 1)
									context = torch.max(context, dim=1)[0] # max_pooling
								else:
									context = torch.zeros([self.hidden_size], dtype=sequence_output.dtype, device=sequence_output.device)
									#print('e1_start:', e1_start, '/ e1_end:', e1_end, '/ e2_start:', e2_start, '/ e2_end:', e2_end)
									#print('ZERO tensor!!! - context.device:', context.device)
									#print('ZERO tensor!!! - context.dtype:', context.dtype)
									#input('enter..')
								
								#print('sequence_output.device:', sequence_output.device)
								#print('e1_rep.device:', e1_rep.device)
								#print('e2_rep.device:', e2_rep.device)
								#print('context.device:', context.device)
								#print('sequence_output.dtype:', sequence_output.dtype)
								#print('e1_rep.dtype:', e1_rep.dtype)
								#print('e2_rep.dtype:', e2_rep.dtype)
								#print('context.dtype:', context.dtype)
								
								e1_e2_plus_context = torch.cat((e1_rep, context))
								e1_e2_plus_context = torch.cat((e1_e2_plus_context, e2_rep))
								
								rel_rep_buffer.append(e1_e2_plus_context)
								rel_label_buffer.append(ppi_label)

								del e1_rep
								del e2_rep
								del context
								del e1_e2_plus_context
							
							
							
						
				
				
				
				
				
				rel_rep_buffer_list.append(rel_rep_buffer)
				rel_label_buffer_list.append(rel_label_buffer)
				
				total_rel_count += len(rel_rep_buffer)
				
				if len(rel_rep_buffer) != len(rel_label_buffer): # debug
					raise ValueError(
						"The representation size (%d) and the label size (%d) are not the same." % (config.hidden_size, config.num_attention_heads)
					)
				
				
				
				
			'''
			print('predictions.requires_grad:', predictions.requires_grad)
			print('input_tokens.requires_grad:', input_tokens.requires_grad)
			print('pred.requires_grad:', pred.requires_grad)
			print('ppi.requires_grad:', ppi.requires_grad)
			input('enter...')
			'''

			if total_rel_count > 0:
				

				
				# the ppi_relations are padded in datacollator, so the size should be the same as the other samples running on diffenent gpus.
				#max_length = len([x for x in torch.split(ppi_relations[0], 5)])
				
				
				'''
				for r_l in rel_rep_buffer_list:
					print(len(r_l))
				print('len(max(rel_rep_buffer_list, key=len)):', len(max(rel_rep_buffer_list, key=len)))
				print('max_length:', max_length)
				input('enter...')
				'''
				
				#max_length = len(max(rel_rep_buffer_list, key=len))
				max_length = 3000
				pad_token = None
				for i in rel_rep_buffer_list:
					for j in i:
						pad_token = torch.zeros_like(j)
						break
					if pad_token is not None:
						break
				rel_rep_buffer_list_padded = [x + [pad_token]*(max_length - len(x)) for x in rel_rep_buffer_list]
				rel_label_buffer_list_padded = [x + [-100]*(max_length - len(x)) for x in rel_label_buffer_list]
				
				'''
				print(rel_rep_buffer_list)
				print(rel_label_buffer_list)
				print(rel_rep_buffer_list_padded)
				print(rel_label_buffer_list_padded)
				'''

				#rel_rep = torch.stack([a for a in rel_rep_buffer], dim=0)

				
				# TODO: add paddings when 'i' is empty. 
				# -> No, it's not necessary. It shouldn't be empty, and if the protein entity combination is not found by NER, then assign a garbage tensor like a zero tensor.
				#    Actually, it's already padded above.
				l = []
				for i in rel_rep_buffer_list_padded:
					l.append(torch.stack([x for x in i], dim=0))
				
				rel_rep = torch.stack([x for x in l], dim=0)
				
				del rel_rep_buffer_list
				del rel_rep_buffer_list_padded

				rel_logits = self.rel_classifier(rel_rep)
				
								
				
				ppi_labels = torch.tensor(rel_label_buffer_list_padded, dtype=torch.long, device=ppi_relations.device)
				del rel_label_buffer_list_padded
				
				
				
				'''
				active_loss = attention_mask.view(-1) == 1
				active_logits = entity_logits.view(-1, self.num_labels)
				active_labels = torch.where(
					active_loss, labels.view(-1), torch.tensor(entity_loss_fct.ignore_index).type_as(labels)
				)
				entity_loss = entity_loss_fct(active_logits, active_labels)
				'''
				
				rel_loss_fct = CrossEntropyLoss()
				
				active_loss = ppi_labels.view(-1) != -100
				active_logits = rel_logits.view(-1, self.num_ppi_labels)
				active_labels = torch.where(
					active_loss, ppi_labels.view(-1), torch.tensor(rel_loss_fct.ignore_index).type_as(ppi_labels)
				)
				
				
				
				'''
				print(rel_rep)
				print(rel_rep.shape)
				print(rel_logits.shape)
				print(active_logits.shape)
				print(ppi_labels)
				print(ppi_labels.shape)
				print(active_labels)
				print(active_labels.shape)
				input('enter...')
				'''

				
				

				rel_loss = rel_loss_fct(active_logits, active_labels)

				# sum the two losses.
				loss = entity_loss + rel_loss
				#loss = rel_loss
			else:
				loss = entity_loss

			logits = rel_logits # TODO: fix this.


			'''	
			print(sequence_output.shape) # ner_classifier input. e.g., torch.Size([8, 84, 768])
			print(entity_logits.shape) # e.g., torch.Size([8, 84, 3])
			print(rel_rep.shape) # e.g., torch.Size([59, 2304]) -> error!!
			print(rel_logits.shape) # e.g., torch.Size([59, 2]) -> error!!
			input('enter...')
			'''
			
			
			
			# temporarily closed to test joint_ner_ppi data. 04-29-2021
			""" 
			# calculate PPI loss
			
			predictions = torch.argmax(entity_logits, dim=2)
			
			true_predictions = [
				[p.item() for (p, l) in zip(prediction, label) if l != -100]
				for prediction, label in zip(predictions, labels)
			]
			true_labels = [
				[l.item() for (p, l) in zip(prediction, label) if l != -100]
				for prediction, label in zip(predictions, labels)
			]
			
			correct_cases = []
			for idx in range(len(true_predictions)):
				if true_predictions[idx] == true_labels[idx]:
					correct_cases.append(idx)

			buffer = []
			
			# loss calculation using Entity START markers. 04-04-2021
			if self.relation_representation == 'EM_entity_start':
				blankv1v2 = sequence_output[:, e1_e2_start, :]
				
				for i in range(blankv1v2.shape[0]): # iterate batch & collect

					if i not in correct_cases:
						continue

					v1v2 = blankv1v2[i, i, :, :]
					v1v2 = torch.cat((v1v2[0], v1v2[1]))
					buffer.append(v1v2)
				del blankv1v2
			
			# loss calculation using mention pooling. 04-25-2021
			elif self.relation_representation in ['STANDARD_mention_pooling', 'EM_mention_pooling']:
				for i in range(sequence_output.shape[0]): # iterate batch & collect
					
					if i not in correct_cases:
						continue
						
					e1_start = entity_mention[i][0]
					e1_end   = entity_mention[i][1]
					e2_start = entity_mention[i][2]
					e2_end   = entity_mention[i][3]
					
					e1_rep = sequence_output[i, e1_start:e1_end, :]
					e2_rep = sequence_output[i, e2_start:e2_end, :]
					
					e1_rep_e2_rep = torch.cat((e1_rep, e2_rep))
					e1_rep_e2_rep = torch.transpose(e1_rep_e2_rep, 0, 1)
					max_pooling = torch.max(e1_rep_e2_rep, dim=1)[0]
					
					buffer.append(max_pooling)

			# loss calculation using CLS token.
			elif self.relation_representation in ['STANDARD_cls_token', 'EM_cls_token']:
				cls_token = sequence_output[:, 0, :]		
				for i in range(cls_token.shape[0]): # iterate batch & collect

					if i not in correct_cases:
						continue
						
					buffer.append(cls_token[i])
				del cls_token

			if len(buffer) > 0:
				rel_rep = torch.stack([a for a in buffer], dim=0)
				del buffer

				#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
				#filtered_ppi_relations = torch.cuda.LongTensor(filtered_ppi_relations)
				#filtered_ppi_relations = torch.tensor(filtered_ppi_relations, dtype=torch.cuda.LongTensor, device=cuda0)
				
				ppi_relations = ppi_relations.view(-1)
				filtered_ppi_relations = torch.index_select(ppi_relations, 0, torch.tensor(correct_cases, dtype=torch.long, device=ppi_relations.device))

				rel_logits = self.rel_classifier(rel_rep)
				rel_loss_fct = CrossEntropyLoss()
				rel_loss = rel_loss_fct(rel_logits, filtered_ppi_relations)
				
				'''
				print('ppi_relations:', ppi_relations)
				print('filtered_ppi_relations:', filtered_ppi_relations)
				print('correct_cases:', correct_cases)
				print('input_ids.device:', input_ids.device)
				print('ppi_relations.device:', ppi_relations.device)
				
				print('rel_rep:', rel_rep)
				print('rel_rep.shape:', rel_rep.shape)
				print('entity_logits:', entity_logits)
				print('entity_logits.shape:', entity_logits.shape)
				print('labels:', labels)
				print('labels.shape:', labels.shape)
				print('active_logits:', active_logits)
				print('active_logits.shape:', active_logits.shape)
				print('active_labels:', active_labels)
				print('active_labels.shape:', active_labels.shape)
				print('self.num_labels:', self.num_labels)
				
				print('rel_logits:', rel_logits)
				print('rel_logits.shape:', rel_logits.shape)
				print('ppi_relations:', ppi_relations)
				print('ppi_relations.shape:', ppi_relations.shape)
				#print('labels.squeeze(1)[:,0]:', labels.squeeze(1)[:,0])
				#input('enter..')
				'''
				
				# sum the two losses.
				loss = entity_loss + rel_loss
			else:
				loss = entity_loss
				
			logits = entity_logits # TODO: fix this.
			"""
			
		else:
			logits = self.classifier(sequence_output)

			loss = None
			if labels is not None:
				loss_fct = CrossEntropyLoss()
				# Only keep active parts of the loss
				if attention_mask is not None:
					active_loss = attention_mask.view(-1) == 1
					active_logits = logits.view(-1, self.num_labels)
					active_labels = torch.where(
						active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
					)
					loss = loss_fct(active_logits, active_labels)
				else:
					loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
		
		if not return_dict:
			output = (logits,) + outputs[2:]
			return ((loss,) + output) if loss is not None else output

		return TokenClassifierOutput(
			loss=loss,
			logits=logits,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
			# [START][GP] - return ppi_labels for PPI classification in the joint learning. 10-01-2021
			ppi_labels=ppi_labels,
			# [END][GP] - return ppi_labels for PPI classification in the joint learning. 10-01-2021
		)



@add_start_docstrings(
	"""
	Bert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
	layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
	""",
	BERT_START_DOCSTRING,
)
class BertForQuestionAnswering(BertPreTrainedModel):

	_keys_to_ignore_on_load_unexpected = [r"pooler"]

	def __init__(self, config):
		super().__init__(config)
		self.num_labels = config.num_labels

		self.bert = BertModel(config, add_pooling_layer=False)
		self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

		self.init_weights()

	@add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
	@add_code_sample_docstrings(
		processor_class=_TOKENIZER_FOR_DOC,
		checkpoint=_CHECKPOINT_FOR_DOC,
		output_type=QuestionAnsweringModelOutput,
		config_class=_CONFIG_FOR_DOC,
	)
	def forward(
		self,
		input_ids=None,
		attention_mask=None,
		token_type_ids=None,
		position_ids=None,
		head_mask=None,
		inputs_embeds=None,
		start_positions=None,
		end_positions=None,
		output_attentions=None,
		output_hidden_states=None,
		return_dict=None,
	):
		r"""
		start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
			Labels for position (index) of the start of the labelled span for computing the token classification loss.
			Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
			sequence are not taken into account for computing the loss.
		end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
			Labels for position (index) of the end of the labelled span for computing the token classification loss.
			Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
			sequence are not taken into account for computing the loss.
		"""
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

		sequence_output = outputs[0]

		logits = self.qa_outputs(sequence_output)
		start_logits, end_logits = logits.split(1, dim=-1)
		start_logits = start_logits.squeeze(-1).contiguous()
		end_logits = end_logits.squeeze(-1).contiguous()

		total_loss = None
		if start_positions is not None and end_positions is not None:
			# If we are on multi-GPU, split add a dimension
			if len(start_positions.size()) > 1:
				start_positions = start_positions.squeeze(-1)
			if len(end_positions.size()) > 1:
				end_positions = end_positions.squeeze(-1)
			# sometimes the start/end positions are outside our model inputs, we ignore these terms
			ignored_index = start_logits.size(1)
			start_positions = start_positions.clamp(0, ignored_index)
			end_positions = end_positions.clamp(0, ignored_index)

			loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
			start_loss = loss_fct(start_logits, start_positions)
			end_loss = loss_fct(end_logits, end_positions)
			total_loss = (start_loss + end_loss) / 2

		if not return_dict:
			output = (start_logits, end_logits) + outputs[2:]
			return ((total_loss,) + output) if total_loss is not None else output

		return QuestionAnsweringModelOutput(
			loss=total_loss,
			start_logits=start_logits,
			end_logits=end_logits,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)
