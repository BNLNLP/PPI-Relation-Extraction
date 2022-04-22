import os
import sys
import logging
import shutil
import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Union, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

import datasets
from datasets import load_dataset, load_metric

import transformers
from transformers import (
	AutoConfig,
	AutoModelForTokenClassification,
	AutoTokenizer,
	BertTokenizerFast,
	RobertaTokenizerFast,
	DebertaV2Tokenizer,
	DebertaTokenizerFast,
	DataCollatorForTokenClassification,
	HfArgumentParser,
	PreTrainedModel,
	PreTrainedTokenizerFast,
	PretrainedConfig,
	Trainer,
	TrainingArguments,
	EvalPrediction,
	set_seed,
)
from transformers.data.data_collator import DataCollator, InputDataClass
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

from dataset_utils import *
from model import BertForRelationClassification, RobertaForRelationClassification
from data_collator import DataCollatorForRelationClassification

### TODO: remove this if not necessary.
# this is used for DDI evaluation. Remove it once DDI evaluation is combined with seqeval.
from sklearn.metrics import f1_score, accuracy_score, classification_report, recall_score, precision_score, precision_recall_fscore_support

### TODO: uncomment this after test.
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
#check_min_version("4.17.0")

### TODO: define requirements by referring the following examples.
#require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/token-classification/requirements.txt")
#require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")


# To avoid the following warning message.
"""
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
		- Avoid using `tokenizers` before the fork if possible
		- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
"""
os.environ["TOKENIZERS_PARALLELISM"] = "true"


dataset_list = ['ChemProt_BLURB', 'DDI_BLURB', 'GAD_BLURB']

dataset_max_seq_length = {
	#"ChemProt_BLURB": 256, # some samples (count: 11) are longer than 256 tokens.
	#"DDI_BLURB": 256, # many samples are longer than 256 tokens. 
	"GAD_BLURB": 128,
}

dataset_special_tokens = {
	"ChemProt_BLURB": ["@GENE$", "@CHEMICAL$", "@CHEM-GENE$"],
	"DDI_BLURB": ["@DRUG$", "@DRUG-DRUG$"],
	"GAD_BLURB": ["@GENE$", "@DISEASE$"],
}

entity_marker_special_tokens = {
	"EM": ["[E1]", "[/E1]", "[E2]", "[/E2]", "[E1-E2]", "[/E1-E2]"],
	"ChemProt_BLURB": ["[GENE]", "[/GENE]", "[CHEM]", "[/CHEM]", "[CHEM-GENE]", "[/CHEM-GENE]"],
	"DDI_BLURB": ["[DRUG]", "[DRUG-DRUG]"],
	"GAD_BLURB": ["[GENE]", "[/GENE]", "[DISEASE]", "[/DISEASE]"],
}

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
	"""
	Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
	"""

	# [START][GP] - input parameter for a list of models. 04-11-2021
	model_name_or_path: str = field(
		default='bert-base-cased',
		metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
	)
	model_list: List[str] = field(
		default_factory=lambda: ['bert-base-cased', 
								 'bert-large-cased', 
								 'dmis-lab/biobert-base-cased-v1.1', 
								 'dmis-lab/biobert-large-cased-v1.1'],
		metadata={"help": "a list of models."},
	)
	# [END][GP] - input parameter for a list of models. 04-11-2021

	config_name: Optional[str] = field(
		default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
	)
	tokenizer_name: Optional[str] = field(
		default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
	)
	cache_dir: Optional[str] = field(
		default=None,
		metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
	)
	model_revision: str = field(
		default="main",
		metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
	)
	use_auth_token: bool = field(
		default=False,
		metadata={
			"help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
			"with private models)."
		},
	)
	# [START][GP] - added do_lower_case parameter for tokenizer. 04-07-2021
	do_lower_case: bool = field(
		default=False,
		metadata={
			"help": "Whether to lowercase words or not. Basically, this option follows model's config, "
			"but, some models (e.g., BioBERT cased) needs to be explicitly set."
		},
	)
	# [END][GP] - added do_lower_case parameter for tokenizer. 04-07-2021


@dataclass
class DataTrainingArguments:
	"""
	Arguments pertaining to what data we are going to input our model for training and eval.
	"""
	
	task_name: Optional[str] = field(default="re", metadata={"help": "The name of the task."})

	dataset_name: Optional[str] = field(
		default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
	)
	dataset_config_name: Optional[str] = field(
		default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
	)
	max_seq_length: int = field(
		default=512,
		metadata={
			"help": "The maximum total input sequence length after tokenization. Sequences longer "
			"than this will be truncated, sequences shorter will be padded."
		},
	)
	train_file: Optional[str] = field(
		default=None, metadata={"help": "The input training data file (a JSON file)."}
	)
	validation_file: Optional[str] = field(
		default=None,
		metadata={"help": "An optional input evaluation data file to evaluate on (a JSON file)."},
	)
	test_file: Optional[str] = field(
		default=None,
		metadata={"help": "An optional input test data file to predict on (a JSON file)."},
	)
	overwrite_cache: bool = field(
		default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
	)
	preprocessing_num_workers: Optional[int] = field(
		default=None,
		metadata={"help": "The number of processes to use for the preprocessing."},
	)
	pad_to_max_length: bool = field(
		default=False,
		metadata={
			"help": "Whether to pad all samples to model maximum sentence length. "
			"If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
			"efficient on GPU but very bad for TPU."
		},
	)
	max_train_samples: Optional[int] = field(
		default=None,
		metadata={
			"help": "For debugging purposes or quicker training, truncate the number of training examples to this "
			"value if set."
		},
	)
	max_eval_samples: Optional[int] = field(
		default=None,
		metadata={
			"help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
			"value if set."
		},
	)
	max_test_samples: Optional[int] = field(
		default=None,
		metadata={
			"help": "For debugging purposes or quicker training, truncate the number of test examples to this "
			"value if set."
		},
	)

	# [START][GP] - data parameters.
	dataset_dir: Optional[str] = field(
		default=None, metadata={"help": "The path to the parent directory of all datasets."},
	)
	relation_types: str = field(
		default=None, metadata={"help": "Relation type file (name and id)."},
	)
	entity_types: str = field(
		default=None, metadata={"help": "Entity type file (name and id)."},
	)
	relation_representation: str = field(
		default='EM_entity_start',
		metadata={"help": "vairous relation representations from [2019] Matching the Blanks: Distributional Similarity for Relation Learning. "
						  "Largely, the representations are divided into standard and entity markers. "
						  "Options: "
						  "1) standard: STANDARD_cls_token, "
						  "				STANDARD_mention_pooling, "
						  "2) entity markers (EM): EM_cls_token, "
						  "						   EM_mention_pooling, "
						  "						   EM_entity_start, "
						  " * for poolings, max pooling is used. "},
	)
	
	use_context: str = field(
		default=None,
		metadata={"help": "Here, context indicates tokens related to entities' relational information. "
						  "The context is appended to relation representations.  "
						  "Options: "
						  "1) attn_based: context based on attention probability calculation, "
						  "2) local: local context (tokens between the two entities) "},
	)
	
	'''
	use_local_context: Optional[bool] = field(
		default=False, 
		metadata={"help": "Whether to use local context (tokens between the two entities)."},
	)
	'''

	use_entity_type_embeddings: Optional[bool] = field(
		default=False, 
		metadata={"help": "Whether to use entity type embeddings."},
	)
	use_entity_typed_marker: Optional[bool] = field(
		default=False, 
		metadata={
			"help": "Whether to use entity typed marker. E.g., [GENE], [/GENE] instead of [E1], [/E1] "
			"This value is used in conjunction with EM representation."
		},
	)
	# [END][GP] - data parameters.

	# [START][GP] - cross-validation parameters.
	do_cross_validation: Optional[bool] = field(
		default=False, 
		metadata={"help": "Whether to use cross-validation for evaluation."}
	)
	num_of_folds: Optional[int] = field(
		default=10, 
		metadata={"help": "The number of folds for the cross-validation."}
	)
	ratio: Optional[str] = field(
		default='80-10-10', 
		metadata={"help": "train/dev/test ratio: 80-10-10, 70-15-15, 60-20-20"}
	)
	# [END][GP] - cross-validation parameters.
	
	save_predictions: Optional[bool] = field(
		default=False, 
		metadata={"help": "Whether to save predictions along with ground truth labels. It's usually for debugging purpose."}
	)
	
	def __post_init__(self):
		if self.dataset_name is None and self.train_file is None and self.validation_file is None:
			raise ValueError("Need either a dataset name or a training/validation file.")
		elif self.dataset_name is not None:
			if self.dataset_name not in dataset_list:
				raise ValueError("Unknown dataset, you should pick one in " + ", ".join(dataset_list))
		else:
			if self.train_file is not None:
				extension = self.train_file.split(".")[-1]
				assert extension == "json", "`train_file` should be a json file."
			if self.validation_file is not None:
				extension = self.validation_file.split(".")[-1]
				assert extension == "json", "`validation_file` should be a json file."
			if self.test_file is not None:
				extension = self.test_file.split(".")[-1]
				assert extension == "json", "`test_file` should be a json file."


def main():
	# See all possible arguments in src/transformers/training_args.py
	# or by passing the --help flag to this script.
	# We now keep distinct sets of args, for a cleaner separation of concerns.

	parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
	if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
		# If we pass only one argument to the script and it's the path to a json file,
		# let's parse it to get our arguments.
		model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
	else:
		model_args, data_args, training_args = parser.parse_args_into_dataclasses()
	
	
	# debugging
	'''
	print('model_args:', model_args)
	print('data_args:', data_args)
	print('training_args:', training_args)
	sys.exit(1)
	'''
	
	smoke_test = False
	

	# Setup logging
	logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
		datefmt="%m/%d/%Y %H:%M:%S",
		handlers=[logging.StreamHandler(sys.stdout)],
	)
	
	log_level = training_args.get_process_log_level()
	logger.setLevel(log_level)
	datasets.utils.logging.set_verbosity(log_level)
	transformers.utils.logging.set_verbosity(log_level)
	transformers.utils.logging.enable_default_handler()
	transformers.utils.logging.enable_explicit_format()

	# Log on each process the small summary:
	logger.warning(
		f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
		+ f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
	)
	logger.info(f"Training/evaluation parameters {training_args}")
	
	relation_representation = data_args.relation_representation
	use_entity_typed_marker = data_args.use_entity_typed_marker
	
	model_list = model_args.model_list
	task_name = data_args.task_name
	dataset_name = data_args.dataset_name
	
	if not os.path.exists(training_args.output_dir):
		os.makedirs(training_args.output_dir)
	
	for model_name in model_list:
		model_args.model_name_or_path = model_name
		training_args.output_dir = os.path.join(training_args.output_dir, model_name)
		output_dir_name = relation_representation
		
		if data_args.use_context != None:
			if data_args.use_context == "attn_based":
				output_dir_name += "_ac"
			elif data_args.use_context == "local":
				output_dir_name += "_lc"
		
		if data_args.use_entity_type_embeddings:
			output_dir_name += "_et"
			
		if data_args.use_entity_typed_marker:
			output_dir_name += "_tm"
		
		training_args.output_dir = os.path.join(training_args.output_dir, output_dir_name)

		# Detecting last checkpoint.
		last_checkpoint = None
		if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
			last_checkpoint = get_last_checkpoint(training_args.output_dir)
			if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
				raise ValueError(
					f"Output directory ({training_args.output_dir}) already exists and is not empty. "
					"Use --overwrite_output_dir to overcome."
				)
			elif last_checkpoint is not None:
				logger.info(
					f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
					"the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
				)
		
		# Set seed before initializing model.
		set_seed(training_args.seed)

		# if not CV, there is only one dataset (train/dev/test).
		if data_args.do_cross_validation:
			num_of_datasets = data_args.num_of_folds
		else:
			num_of_datasets = 1
		
		relation_type_file = os.path.join(data_args.dataset_dir, dataset_name)
		relation_type_file = os.path.join(relation_type_file, "relation_types.json")
		relation_types = json.load(open(relation_type_file))
		
		entity_type_file = os.path.join(data_args.dataset_dir, dataset_name)
		entity_type_file = os.path.join(entity_type_file, "entity_types.json")
		entity_types = json.load(open(entity_type_file))	
		
		label_list = list(relation_types.keys())
		num_labels = len(label_list)

		# Load pretrained model and tokenizer
		#
		# In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
		# download model & vocab.
		config = AutoConfig.from_pretrained(
			model_args.config_name if model_args.config_name else model_args.model_name_or_path,
			num_labels=num_labels,
			finetuning_task=data_args.task_name,
			cache_dir=model_args.cache_dir,
			revision=model_args.model_revision,
			use_auth_token=True if model_args.use_auth_token else None,
			
			# get attention outputs for attention based context.
			# ref: https://discuss.huggingface.co/t/output-attention-true-after-downloading-a-model/907
			output_attentions=True if data_args.use_context == "attn_based" else None,
		)

		# Explicitly set 'do_lower_case' since some models have a wrong case setting. (e.g., BioBERT, SciBERT)
		# HuggingFace ALBERT models and PubMedBERT are uncased.
		if config.model_type == "albert" or "PubMedBERT" in model_name:
			do_lower_case = True 
		else:
			do_lower_case = model_args.do_lower_case
		
		tokenizer_name_or_path = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
		# Use 'add_prefix_space' for GPT2, RoBERTa, DeBERTa
		if config.model_type in {"gpt2", "roberta", "deberta", "deberta-v2"}:
			tokenizer = AutoTokenizer.from_pretrained(
				tokenizer_name_or_path,
				cache_dir=model_args.cache_dir,
				use_fast=True,
				revision=model_args.model_revision,
				use_auth_token=True if model_args.use_auth_token else None,
				add_prefix_space=True,
			)
		else:
			# Set 'do_lower_case' when loading a tokenizer. It's not working to change the variable after it's loaded (i.e., tokenizer.do_lower_case = False).
			# GPT2, RoBERTa, DeBERTa don't have 'do_lower_case'. 
			tokenizer = AutoTokenizer.from_pretrained(
				tokenizer_name_or_path,
				cache_dir=model_args.cache_dir,
				use_fast=True,
				revision=model_args.model_revision,
				use_auth_token=True if model_args.use_auth_token else None,
				do_lower_case=do_lower_case,
			)

		max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
		
		### TODO: delete this if not necessary (it doesn't affect the performance). 03-02-2022
		#
		# The default max length of BioBERT & DeBERTa is too large (1000000000000000019884624838656), so set it to 512.
		'''
		if isinstance(tokenizer, BertTokenizerFast) or isinstance(tokenizer, DebertaTokenizerFast) or isinstance(tokenizer, DebertaV2Tokenizer):
			max_seq_length = 512 
		'''
				
		if dataset_name in dataset_max_seq_length.keys():
			max_seq_length = dataset_max_seq_length[dataset_name]
			
		# Add special tokens for entity markers.
		#
		# Special tokens do not need to be lowercased for uncased models because they are basically all uppercased in the datasets. 03/30/2022
		# When special_tokens is set to False (default) in add_tokens(), it treats added tokens as normal tokens.
		# E.g., (added tokens: '@gene$', '@disease$') tokenizer.tokenize("@gene$, @disease$"), tokenizer.tokenize("@GENE$, @DISEASE$")
		#       -> the same output: ['@gene$', '@disease$']
		# When special_tokens is set to True in add_tokens(), it treats added tokens as special tokens.
		# E.g., (added tokens: '@gene$', '@disease$') tokenizer.tokenize("@gene$, @disease$"), tokenizer.tokenize("@GENE$, @DISEASE$")
		#       -> the different output: ['@gene$', '@disease$'], ['@', 'gene', '$', ',', '@', 'disease', '$']
		#
		# For some reason, add_tokens(special_tokens=True) doesn't update tokenizer.all_special_ids, tokenizer.all_special_tokens, ... 03/30/2022
		# Use add_special_tokens() instead.
		if relation_representation.startswith('EM'):
			additional_special_tokens = {"additional_special_tokens": entity_marker_special_tokens[dataset_name] if use_entity_typed_marker \
																	  else entity_marker_special_tokens['EM']}
			tokenizer.add_special_tokens(additional_special_tokens)
		
		# Add the special tokens that are used to replace entity names (entity anonymization or dummification). E.g,. ChemProt, DDI, GAD
		if dataset_name in dataset_special_tokens.keys():
			additional_special_tokens = {"additional_special_tokens": dataset_special_tokens[dataset_name]}
			tokenizer.add_special_tokens(additional_special_tokens)

		# debugging
		'''
		#save_as_pickle("%s_tokenizer.pkl" % model_name, tokenizer_dict[task_name])
		#logger.info("Saved %s tokenizer at ./data/%s_tokenizer.pkl" %(model_name, model_name))
		
		#e1_id = tokenizer_dict[task_name].convert_tokens_to_ids('[E1]')
		#e2_id = tokenizer_dict[task_name].convert_tokens_to_ids('[E2]')
		#assert e1_id != e2_id != 1
		
		print(config.model_type)
		print(do_lower_case)
		#print(tokenizer.do_lower_case)
		input('enter..')

		print(tokenizer)
		#print(isinstance(tokenizer, BertTokenizerFast))
		print('tokenizer.model_max_length:', tokenizer.model_max_length)
		print(tokenizer.tokenize("[E1]BioBERT[/E1] sets [E2]do_lower_case[/E2] to True by default. Test IT."))
		input('enter..')
		'''	

		
		def get_model():
			
			
			
			#print(config.vocab_size)
			
			config = AutoConfig.from_pretrained(
				model_args.config_name if model_args.config_name else model_args.model_name_or_path,
				num_labels=num_labels,
				finetuning_task=data_args.task_name,
				cache_dir=model_args.cache_dir,
				revision=model_args.model_revision,
				use_auth_token=True if model_args.use_auth_token else None,
				
				# get attention outputs for attention based context.
				# ref: https://discuss.huggingface.co/t/output-attention-true-after-downloading-a-model/907
				output_attentions=True if data_args.use_context == "attn_based" else None,
			)
			
			
				
			
			### TODO: make it cleaner by creating 'AutoModelForRelationClassification'.
			if config.model_type == "bert":
				model = BertForRelationClassification.from_pretrained(
					model_args.model_name_or_path,
					from_tf=bool(".ckpt" in model_args.model_name_or_path),
					config=config,
					cache_dir=model_args.cache_dir,
					revision=model_args.model_revision,
					use_auth_token=True if model_args.use_auth_token else None,
					
					# keyword parameters for RE
					relation_representation=relation_representation,
					use_context=data_args.use_context,
					use_entity_type_embeddings=data_args.use_entity_type_embeddings,
					num_entity_types=len(entity_types),
					tokenizer=tokenizer,
					#ignore_mismatched_sizes=True
				)
			elif config.model_type == "roberta":
				model = RobertaForRelationClassification.from_pretrained(
					model_args.model_name_or_path,
					from_tf=bool(".ckpt" in model_args.model_name_or_path),
					config=config,
					cache_dir=model_args.cache_dir,
					revision=model_args.model_revision,
					use_auth_token=True if model_args.use_auth_token else None,
					
					# keyword parameters for RE
					relation_representation=relation_representation,
					use_context=data_args.use_context,
					use_entity_type_embeddings=data_args.use_entity_type_embeddings,
					num_entity_types=len(entity_types),
					tokenizer=tokenizer,
					#ignore_mismatched_sizes=True
				)
			
			
			
			#print(config)
			#input('etner..')
			
			
			
			# Resize input token embeddings matrix of the model since new tokens have been added.
			# this funct is used if the number of tokens in tokenizer is different from config.vocab_size.
			model.resize_token_embeddings(len(tokenizer))
			
			
			#print('len(tokenizer):', len(tokenizer))
			#print('model.get_input_embeddings().num_embeddings:', model.get_input_embeddings().num_embeddings)
			#input('enter..')
			
			return model


		# Padding strategy
		if data_args.pad_to_max_length:
			padding = "max_length"
		else:
			# We will pad later in data collator, dynamically at batch creation, to the max sequence length in each batch
			padding = False
		
		def compute_metrics(p: EvalPrediction):
			pred, true = p.predictions, p.label_ids
			
			'''
			print(pred)
			print(true)
			print(pred.shape)
			print(true.shape)
			'''
			
			pred = np.argmax(pred, axis=1)
			true = true.flatten()

			pred = pred.tolist()
			true = true.tolist()
			
			#pred = [label_list[x] for x in pred]
			#true = [label_list[x] for x in true]
			
			#print(pred)
			#print(true)
			#input('enter..')
			
			
			# Remove ignored labels.
			# For ChemProt, ignore false labels. "CPR:false": "id": 0
			# For DDI, ignore false labels. "DDI-false": "id": 0
			# For TACRED, ignore no relation labels. "no_relation": "id": 0
			if any(x == dataset_name for x in ['ChemProt_BLURB', 'DDI_BLURB', 'TACRED']):
				cleaned_pred_true = [(p, t) for (p, t) in zip(pred, true) if t != 0]
				pred = [x[0] for x in cleaned_pred_true]
				true = [x[1] for x in cleaned_pred_true]
			
			# metrics ref: https://github.com/huggingface/datasets/tree/master/metrics
			a_m = load_metric("accuracy")
			p_m = load_metric("precision")
			r_m = load_metric("recall")
			f_m = load_metric("f1")

			a = a_m.compute(predictions=pred, references=true)
			p = p_m.compute(predictions=pred, references=true, average="micro")
			r = r_m.compute(predictions=pred, references=true, average="micro")
			f = f_m.compute(predictions=pred, references=true, average="micro")
			
			return {"accuracy": a["accuracy"], "precision": p["precision"], "recall": r["recall"], "f1": f["f1"]}
		
		for dataset_num in range(num_of_datasets):
			logger.info("\n\n*** Dataset number: " + str(dataset_num) + " ***\n\n")

			# Loading a dataset from your local files.
			data_files = read_dataset(dataset_num, task_name, data_args)

			dataset = featurize_data(data_files, tokenizer, padding, max_seq_length, relation_representation, use_entity_typed_marker)

			train_dataset = dataset["train"]
			eval_dataset = dataset["validation"] if "validation" in dataset else None
			test_dataset = dataset["test"]
			
			# Data collator
			data_collator = DataCollatorForRelationClassification(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)

			#print(training_args)
			
			
			# Evaluate during training and a bit more often
			# than the default to be able to prune bad trials early.
			# Disabling tqdm is a matter of preference.
			#training_args = TrainingArguments(
			#	"test", evaluation_strategy="steps", eval_steps=500, disable_tqdm=True)
			
			#training_args.evaluation_strategy = "steps"
			#training_args.eval_steps = 500
			#training_args.disable_tqdm = True
			
			#print(training_args)
			#input('enter...')


			# Initialize our Trainer
			trainer = Trainer(
				#model=model,
				model_init=get_model,
				args=training_args,
				train_dataset=train_dataset if training_args.do_train else None,
				eval_dataset=eval_dataset if training_args.do_eval else None,
				tokenizer=tokenizer,
				data_collator=data_collator,
				compute_metrics=compute_metrics,
			)
			
			
			'''
			# Add the special tokens that are used to replace entity names (entity anonymization or dummification). E.g,. ChemProt, DDI, GAD
			if dataset_name in dataset_special_tokens.keys():
				special_tokens = list(map(lambda x: x.lower(), dataset_special_tokens[dataset_name])) if do_lower_case else \
								 dataset_special_tokens[dataset_name]
				
			trainer.tokenizer.add_tokens(special_tokens)
			trainer.model.resize_token_embeddings(len(trainer.tokenizer))
			'''
			

			#print('len(trainer.tokenizer):', len(trainer.tokenizer))
			#input('enter..')
			
			#print(torch.cuda.device_count())
			#print(os.cpu_count())
			#input('enter..')
			
			
			
			# resources_per_trial={"cpu": 1, "gpu": 1}
			
			
			#best_run = trainer.hyperparameter_search(n_trials=100, compute_objective='accuracy', direction="maximize", backend='ray',
			#							 			 search_alg=HyperOptSearch(metric='accuracy', mode='max', use_early_stopped_trials=True), scheduler=pbt,
			#										 keep_checkpoints_num=1)
			
			
			
			# ref:
			# https://github.com/ray-project/ray/blob/65d72dbd9148b725761f733559e3c5c72f15da9a/python/ray/tune/examples/pbt_transformers/pbt_transformers.py#L12
			# https://discuss.huggingface.co/t/using-hyperparameter-search-in-trainer/785/55
			# https://docs.ray.io/en/latest/tune/api_docs/schedulers.html#tune-scheduler-pbt
			# https://github.com/ray-project/ray/issues/14108
			# https://discuss.huggingface.co/t/using-hyperparameter-search-in-trainer/785/29?page=2
			# https://discuss.huggingface.co/t/using-hyperparameter-search-in-trainer/785/10?page=2
			# https://github.com/huggingface/transformers/pull/6576
			
			from ray.tune.suggest.hyperopt import HyperOptSearch
			from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
			from ray import tune
			from ray.tune import CLIReporter
			import random
			
			# The example of using HyperOptSearch with ASHAScheduler: https://huggingface.co/blog/ray-tune
			hyperopt_search = HyperOptSearch(
				metric="eval_f1", 
				mode="max", 
				use_early_stopped_trials=True
			)
			
			# ref: https://machinelearningmastery.com/how-to-reduce-overfitting-in-deep-learning-with-weight-regularization/
			# The most common type of regularization is L2, also called simply “weight decay,” with values often on a logarithmic scale between 0 and 0.1, such as 0.1, 0.001, 0.0001, etc.
			# Reasonable values of lambda [regularization hyperparameter] range between 0 and 0.1.
			pbt_scheduler = PopulationBasedTraining(
				#time_attr='time_total_s', # "training_iteration",
				metric="eval_f1", # 'mean_accuracy',
				mode="max",
				perturbation_interval=1, # 600.0, 2,
				hyperparam_mutations={
					"weight_decay": [0.0, 0.01], # [0.0, 0.1, 0.01, 0.001, 0.0001, 0.00001], tune.uniform(0.0, 0.01), tune.uniform(0.0, 0.3), lambda: uniform(0.0, 0.3),
					"warmup_ratio": [0.0, 0.1],
					#"warmup_steps":lambda: randint(0, 500),
					"learning_rate": [1e-5, 2e-5, 3e-5, 4e-5, 5e-5], # tune.uniform(1e-5, 5e-5), tune.loguniform(1e-4, 1e-2), tune.uniform(1e-5, 7e-5), lambda: uniform(1e-5, 5e-5),
					"per_device_train_batch_size": [8, 16], # tune.choice([8, 16, 32, 64, 128]), #"per_gpu_train_batch_size": [16, 32, 64],
					#"alpha": lambda: random.uniform(0.0, 1.0),
					"seed": tune.choice(range(1, 20001)), # tune.uniform(1, 20000), tune.choice(range(1, 41))
					"num_train_epochs": tune.choice(range(2, 21)), # tune.randint(2, 21), tune.choice([2, 5, 10, 15, 20]), [2, 5, 10, 15, 20],
				}
			)
			
			tune_config = {
				"per_device_eval_batch_size": 32,
				"max_steps": 1 if smoke_test else -1,  # Used for smoke test.
			}
			
			reporter = CLIReporter(
				parameter_columns={
					"weight_decay": "w_decay",
					"warmup_ratio": "w_ratio",
					"learning_rate": "lr",
					"per_device_train_batch_size": "train_bs/gpu",
					#"per_device_eval_batch_size": "eval_bs/gpu",
					"seed": "seed",
					"num_train_epochs": "num_epochs",
				},
				metric_columns=["eval_f1", "eval_loss"], # "epoch", "training_iteration"
			)
			
			def compute_objective(metrics):
				return metrics["eval_f1"]

			#print(os.cpu_count())
			#print(torch.cuda.device_count())
			#print(torch.cuda.get_device_name(0))
			#print(torch.cuda.is_available())
			
			n_cpu = os.cpu_count()
			n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0

			# Default objective is the sum of all metrics
			# when metrics are provided, so we have to maximize it.
			trainer.hyperparameter_search(
				hp_space=lambda _: tune_config,
				compute_objective=compute_objective,
				direction="maximize", 
				backend="ray", 
				n_trials=20, # number of trials
				
				#search_alg=hyperopt_search,
				scheduler=pbt_scheduler,
				keep_checkpoints_num=1,
				checkpoint_score_attr="training_iteration",
				stop={"training_iteration": 1} if smoke_test else None, # {"training_iteration": 10, "mean_accuracy": 0.98}
				resources_per_trial={"cpu": n_cpu, "gpu": n_gpu},
				progress_reporter=reporter,
				#local_dir="~/ray_results/",
				#name="tune_transformer_pbt",
				log_to_file=True,
				#resume=True,
			)

		logger.info("*** Data iterations are done.  ***")


if __name__ == "__main__":
	main()
	
	