#
# ref: https://colab.research.google.com/github/zphang/zphang.github.io/blob/master/files/notebooks/Multi_task_Training_with_Transformers_NLP.ipynb#scrollTo=U4YUxdIZz3_i
#

import numpy as np
import torch
import torch.nn as nn
import transformers
#import nlp
import dataclasses
import shutil
from torch.utils.data.dataloader import DataLoader
#from transformers.training_args import is_tpu_available  # ImportError: cannot import name 'is_tpu_available' from 'transformers.training_args'
#from transformers.trainer import get_tpu_sampler
from transformers.data.data_collator import DataCollator, InputDataClass
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from typing import List, Union, Dict, Optional, Tuple


## [start] from NER
import os
import sys
from dataclasses import dataclass, field

from datasets import ClassLabel, load_dataset, load_metric

from transformers import (
	AutoConfig,
	AutoModelForTokenClassification,
	AutoTokenizer,
	BertTokenizerFast,
	DataCollatorForTokenClassification,
	HfArgumentParser,
	PreTrainedModel,
	PreTrainedTokenizerFast,
	PretrainedConfig,
	Trainer,
	TrainingArguments,
	set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.5.0.dev0")

## [end] from NER


# PPI
#from transformers_modified.models.bert.modeling_bert import BertForTokenClassification, BertConfig
# PPI

from dataset_utils import *


import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultitaskModel(PreTrainedModel):
	def __init__(self, encoder, taskmodels_dict):
		"""
		Setting MultitaskModel up as a PretrainedModel allows us
		to take better advantage of Trainer features
		"""
		super().__init__(PretrainedConfig())

		self.encoder = encoder

		self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)
		
				
		'''
		print('type(self.encoder):', type(self.encoder))
		
		for k, v in self.taskmodels_dict.items():
			print(k, v)
		
		input('enter..')
		'''
		

	@classmethod
	def create(cls, model_name, model_args, model_type_dict, model_config_dict, data_args, tokenizer_dict):
		"""
		This creates a MultitaskModel using the model class and config objects
		from single-task models. 

		We do this by creating each single-task model, and having them share
		the same encoder transformer.
		"""
		shared_encoder = None
		taskmodels_dict = {}
		for task_name, model_type in model_type_dict.items():
			model = model_type.from_pretrained(
				model_name, 
				config=model_config_dict[task_name],

				from_tf=bool(".ckpt" in model_args.model_name_or_path),
				cache_dir=model_args.cache_dir,
				revision=model_args.model_revision,
				use_auth_token=True if model_args.use_auth_token else None,
				
				relation_representation=data_args.relation_representation,
				num_ppi_labels=len(data_args.ppi_classes),
				tokenizer=tokenizer_dict[task_name],

			)
			if shared_encoder is None:
				shared_encoder = getattr(model, cls.get_encoder_attr_name(model))
			else:
				setattr(model, cls.get_encoder_attr_name(model), shared_encoder)
			taskmodels_dict[task_name] = model
		return cls(encoder=shared_encoder, taskmodels_dict=taskmodels_dict)

	@classmethod
	def get_encoder_attr_name(cls, model):
		"""
		The encoder transformer is named differently in each model "architecture".
		This method lets us get the name of the encoder attribute
		"""
		model_class_name = model.__class__.__name__
		if model_class_name.startswith("Bert"):
			return "bert"
		elif model_class_name.startswith("Roberta"):
			return "roberta"
		elif model_class_name.startswith("Albert"):
			return "albert"
		else:
			raise KeyError(f"Add support for new model {model_class_name}")

	def forward(self, task_name, **kwargs):
		return self.taskmodels_dict[task_name](**kwargs)

'''
class NLPDataCollator(DataCollator):
	"""
	Extending the existing DataCollator to work with NLP dataset batches
	"""
	def collate_batch(self, features: List[Union[InputDataClass, Dict]]) -> Dict[str, torch.Tensor]:
		first = features[0]
		if isinstance(first, dict):
		  # NLP data sets current works presents features as lists of dictionary
		  # (one per example), so we will adapt the collate_batch logic for that
		  if "labels" in first and first["labels"] is not None:
			  if first["labels"].dtype == torch.int64:
				  labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
			  else:
				  labels = torch.tensor([f["labels"] for f in features], dtype=torch.float)
			  batch = {"labels": labels}
		  for k, v in first.items():
			  if k != "labels" and v is not None and not isinstance(v, str):
				  batch[k] = torch.stack([f[k] for f in features])
		  return batch
		else:
		  # otherwise, revert to using the default collate_batch
		  return DefaultDataCollator().collate_batch(features)
'''

class StrIgnoreDevice(str):
	"""
	This is a hack. The Trainer is going call .to(device) on every input
	value, but we need to pass in an additional `task_name` string.
	This prevents it from throwing an error
	"""
	def to(self, device):
		return self


class DataLoaderWithTaskname:
	"""
	Wrapper around a DataLoader to also yield a task name
	"""
	def __init__(self, task_name, data_loader):
		self.task_name = task_name
		self.data_loader = data_loader

		self.batch_size = data_loader.batch_size
		self.dataset = data_loader.dataset

	def __len__(self):
		return len(self.data_loader)
	
	def __iter__(self):
		for batch in self.data_loader:
			batch["task_name"] = StrIgnoreDevice(self.task_name)
			yield batch


class MultitaskDataloader:
	"""
	Data loader that combines and samples from multiple single-task
	data loaders.
	"""
	def __init__(self, dataloader_dict):
		self.dataloader_dict = dataloader_dict
		self.num_batches_dict = {
			task_name: len(dataloader) 
			for task_name, dataloader in self.dataloader_dict.items()
		}
		self.task_name_list = list(self.dataloader_dict)
		self.dataset = [None] * sum(
			len(dataloader.dataset) 
			for dataloader in self.dataloader_dict.values()
		)

	def __len__(self):
		return sum(self.num_batches_dict.values())

	def __iter__(self):
		"""
		For each batch, sample a task, and yield a batch from the respective
		task Dataloader.

		We use size-proportional sampling, but you could easily modify this
		to sample from some-other distribution.
		"""
		task_choice_list = []
		for i, task_name in enumerate(self.task_name_list):
			task_choice_list += [i] * self.num_batches_dict[task_name]
		task_choice_list = np.array(task_choice_list)
		np.random.shuffle(task_choice_list)
		dataloader_iter_dict = {
			task_name: iter(dataloader)
			for task_name, dataloader in self.dataloader_dict.items()
		}
		for task_choice in task_choice_list:
			task_name = self.task_name_list[task_choice]
			yield next(dataloader_iter_dict[task_name])


class MultitaskTrainer(Trainer):

	def get_single_train_dataloader(self, task_name, train_dataset):
		"""
		Create a single-task data loader that also yields task names
		"""
		if self.train_dataset is None:
			raise ValueError("Trainer: training requires a train_dataset.")
		#if is_tpu_available():
		#	train_sampler = get_tpu_sampler(train_dataset)
		else:
			train_sampler = (
				RandomSampler(train_dataset)
				if self.args.local_rank == -1
				else DistributedSampler(train_dataset)
			)

		data_loader = DataLoaderWithTaskname(
			task_name=task_name,
			data_loader=DataLoader(
				train_dataset,
				batch_size=self.args.train_batch_size,
				sampler=train_sampler,
				#collate_fn=self.data_collator.collate_batch,
				collate_fn=self.data_collator,
			),
		)

		'''
		# Trainer class -> train() -> get_train_dataloader()
		def get_train_dataloader(self) -> DataLoader:
			"""
			Returns the training :class:`~torch.utils.data.DataLoader`.

			Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
			to distributed training if necessary) otherwise.

			Subclass and override this method if you want to inject some custom behavior.
			"""
			if self.train_dataset is None:
				raise ValueError("Trainer: training requires a train_dataset.")
			train_sampler = self._get_train_sampler()

			return DataLoader(
				self.train_dataset,
				batch_size=self.args.train_batch_size,
				sampler=train_sampler,
				collate_fn=self.data_collator,
				drop_last=self.args.dataloader_drop_last,
				num_workers=self.args.dataloader_num_workers,
				pin_memory=self.args.dataloader_pin_memory,
			)
		'''

		#if is_tpu_available():
		#	data_loader = pl.ParallelLoader(
		#		data_loader, [self.args.device]
		#	).per_device_loader(self.args.device)
		
		return data_loader

	def get_train_dataloader(self):
		"""
		Returns a MultitaskDataloader, which is not actually a Dataloader
		but an iterable that returns a generator that samples from each 
		task Dataloader
		"""
		return MultitaskDataloader({
			task_name: self.get_single_train_dataloader(task_name, task_dataset)
			for task_name, task_dataset in self.train_dataset.items()
		})
	
	'''
	def get_single_eval_dataloader(self, task_name, eval_dataset):
		if self.eval_dataset is None:
			raise ValueError("Trainer: training requires a eval_dataset.")
		else:
			eval_sampler = (
				RandomSampler(eval_dataset)
				if self.args.local_rank == -1
				else DistributedSampler(eval_dataset)
			)
		
		
		print('self.args.train_batch_size:', self.args.train_batch_size)
		print('self.args.eval_batch_size:', self.args.eval_batch_size)
		
		
		data_loader = DataLoaderWithTaskname(
			task_name=task_name,
			data_loader=DataLoader(
				eval_dataset,
				batch_size=self.args.eval_batch_size,
				sampler=eval_sampler,
				#collate_fn=self.data_collator.collate_batch,
				collate_fn=self.data_collator,
			),
		)

		return data_loader
	'''
	
	'''
	def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
		
		if self.eval_dataset['ner'] is None:
			raise ValueError("Trainer: training requires a eval_dataset.")
		else:
			eval_sampler = (
				RandomSampler(self.eval_dataset['ner'])
				if self.args.local_rank == -1
				else DistributedSampler(self.eval_dataset['ner'])
			)
			
		return DataLoaderWithTaskname(
			task_name='ner',
			data_loader=DataLoader(
				self.eval_dataset['ner'],
				batch_size=self.args.eval_batch_size,
				sampler=eval_sampler,
				#collate_fn=self.data_collator.collate_batch,
				collate_fn=self.data_collator,
			),
		)
	'''
	
## [start] from NER


@dataclass
class ModelArguments:
	"""
	Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
	"""

	# [START][GP] - input parameter for a list of models. - 04-11-2021
	model_name_or_path: str = field(
		default='bert-base-cased',
		metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
	)
	model_list: List[str] = field(
		default_factory=lambda: ['bert-base-cased', 'bert-large-cased', 'dmis-lab/biobert-base-cased-v1.1', 'dmis-lab/biobert-large-cased-v1.1'],
		metadata={"help": "a list of models."},
	)
	# [END][GP] - input parameter for a list of models. - 04-11-2021

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
	
	# [START][GP] - input parameter for a list of tasks. - 04-07-2021
	#task_name: Optional[str] = field(default="ner", metadata={"help": "The name of the task (ner, pos...)."})
	task_list: List[str] = field(
		default_factory=lambda: ['ner', 'ppi', 'joint-ner-ppi'],
		metadata={"help": "a list of tasks."},
	)
	# [END][GP] - input parameter for a list of tasks. - 04-07-2021
	
	dataset_name: Optional[str] = field(
		default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
	)
	dataset_config_name: Optional[str] = field(
		default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
	)
	train_file: Optional[str] = field(
		default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
	)
	validation_file: Optional[str] = field(
		default=None,
		metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
	)
	test_file: Optional[str] = field(
		default=None,
		metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
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
	max_val_samples: Optional[int] = field(
		default=None,
		metadata={
			"help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
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
	label_all_tokens: bool = field(
		default=False,
		metadata={
			"help": "Whether to put the label for one word on all tokens of generated by that word or just on the "
			"one (in which case the other tokens will have a padding index)."
		},
	)
	return_entity_level_metrics: bool = field(
		default=False,
		metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
	)

	
	# [START][GP] - data parameters.
	ner_data: Optional[str] = field(
		default=None, metadata={"help": "The input NER data file."}
	)
	ppi_data: Optional[str] = field(
		default=None, metadata={"help": "The input PPI data file."}
	)
	joint_ner_ppi_data: Optional[str] = field(
		default=None, metadata={"help": "The input joint NER and PPI data file."}
	)
	do_fine_tune: Optional[bool] = field(
		default=True, 
		metadata={"help": "Whether to fine-tune a model for each task using a task-specific data \
						   This is a step after multi-task learning, which replaces a task head with a new one. \
						   This option is only needed for multi-task learning."}
	)
	# explicitly define classes since a dataset may not contain all classes in which case Relation_Mapper assigns overlapped relation ids.
	ppi_classes: List[str] = field(
		default_factory=lambda: ['enzyme', 'structural', 'negative'],
		metadata={"help": "a list of PPI classes."},
	)
	relation_representation: str = field(
		default='EM_entity_start',
		metadata={"help": "vairous relation representations from [2019] Matching the Blanks: Distributional Similarity for Relation Learning. \
						   Largely, architectures are divided into standard and entity markers. \
						   Options: \
						   1) standard: STANDARD_cls_token, STANDARD_mention_pooling, STANDARD_mention_pooling_plus_context\
						   2) entity markers (EM): EM_cls_token, EM_mention_pooling, EM_entity_start, EM_entity_start_plus_context \
						   - multiple relations: Multiple_Relations \
						   * for poolings, max pooling is used. "}
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
	
	save_misclassified_samples: Optional[bool] = field(
		default=False, 
		metadata={"help": "Whether to save misclassified samples for debugging purpose."}
	)
	
	
	'''
	def __post_init__(self):
		if self.dataset_name is None and self.train_file is None and self.validation_file is None:
			raise ValueError("Need either a dataset name or a training/validation file.")
		else:
			if self.train_file is not None:
				extension = self.train_file.split(".")[-1]
				#assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
			if self.validation_file is not None:
				extension = self.validation_file.split(".")[-1]
				#assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
		self.task_name = self.task_name.lower()
	'''
	
## [end] from NER



## [start] from PPI

def evaluate_(output, labels, ignore_idx):
	### ignore index 0 (padding) when calculating accuracy
	idxs = (labels != ignore_idx).squeeze()
	
	o_labels = torch.softmax(output, dim=1).max(1)[1]

	l = labels.squeeze()[idxs]; o = o_labels[idxs]

	# [GP][START] - check tensor size for TypeError: len() of a 0-d tensor.
	if bool(idxs.size()) and len(idxs) > 1:
	# [GP][END] - check tensor size for TypeError: len() of a 0-d tensor.
		acc = (l == o).sum().item()/len(idxs)
	else:
		acc = (l == o).sum().item()
	l = l.cpu().numpy().tolist() if l.is_cuda else l.numpy().tolist()
	o = o.cpu().numpy().tolist() if o.is_cuda else o.numpy().tolist()

	return acc, (o, l)


def evaluate_results(net, test_loader, pad_id, cuda):
	logger.info("Evaluating test samples...")
	acc = 0; out_labels = []; true_labels = []
	net.eval()
	with torch.no_grad():
		for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
			x, e1_e2_start, labels, _,_,_ = data
			attention_mask = (x != pad_id).float()
			token_type_ids = torch.zeros((x.shape[0], x.shape[1])).long()

			if cuda:
				x = x.cuda()
				labels = labels.cuda()
				attention_mask = attention_mask.cuda()
				token_type_ids = token_type_ids.cuda()

			classification_logits = net(x, token_type_ids=token_type_ids, attention_mask=attention_mask, Q=None,\
										e1_e2_start=e1_e2_start)

			accuracy, (o, l) = evaluate_(classification_logits, labels, cuda, ignore_idx=-1)
			out_labels.append([str(i) for i in o]); true_labels.append([str(i) for i in l])
			acc += accuracy
			
			'''
			print('accuracy:', accuracy)
			print('acc:', acc)
			print('o:', o)
			print('l:', l)
			print('out_labels:', out_labels)
			print('true_labels:', true_labels)
			
			#input('enter...')
			'''	

	accuracy = acc/(i + 1)
	
	# [GP][START] - use sklearn metrics for evaluation.
	'''
	results = {
		"accuracy": accuracy,
		"precision": precision_score(true_labels, out_labels),
		"recall": recall_score(true_labels, out_labels),
		"f1": f1_score(true_labels, out_labels)
	}
	'''
	
	from sklearn.metrics import precision_recall_fscore_support
	
	'''
	print('true_labels:', true_labels)
	print('out_labels:', out_labels)
	print('accuracy:', accuracy)
	print('precision:', precision_score(true_labels_one_d, out_labels_one_d))
	print('recall:', recall_score(true_labels_one_d, out_labels_one_d))
	print('f1_score:', f1_score(true_labels_one_d, out_labels_one_d))
	'''
	true_labels_flattened = np.concatenate(true_labels)
	out_labels_flattened = np.concatenate(out_labels)
	#true_labels_flattened = list(true_labels_flattened)
	#out_labels_flattened = list(out_labels_flattened)

	precision, recall, f1, _ = precision_recall_fscore_support(true_labels_flattened, out_labels_flattened, average='weighted')
	results = {
		"accuracy": accuracy,
		"precision": precision,
		"recall": recall,
		"f1": f1
	}
	# [GP][END] - use sklearn metrics for evaluation.
	
	
	logger.info("***** Eval results *****")
	for key in sorted(results.keys()):
		logger.info("  %s = %s", key, str(results[key]))

	return results
	
## [end] from PPI	


def save_results(results_dict, data_list, task_list, label_list, output_dir, do_cross_validation, save_misclassified_samples):
	
	if do_cross_validation:
		all_results_per_data_and_task = {}
		for data_name in data_list: # eval or pred
			all_results_per_data_and_task[data_name] = {}
			for task_name in task_list:
				all_results_per_data_and_task[data_name][task_name] = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
	
	if save_misclassified_samples:
		miscls_samples_per_data_and_task = {}
		for data_name in data_list: # eval or pred
			miscls_samples_per_data_and_task[data_name] = {}
			for task_name in task_list:
				miscls_samples_per_data_and_task[data_name][task_name] = {}
	
	if len(task_list) > 1:
		learning_type = 'mtl' # multi-task learning
	else:
		learning_type = 'stl' # single task learning
	
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
				
	#mis_sents = [] # debug
	
	
	for dataset_num, results in results_dict.items():
		for result in results:
			data_name = result['data_name'] # eval or pred
			preds_dict = result['preds_dict']
			datasets_dict = result['datasets_dict']
			tokenizer_dict = result['tokenizer_dict']
	
			for task_name, pred_output in preds_dict.items():
				if task_name == 'ner':
								
					## [start] from NER
					# Metrics
					metric = load_metric("seqeval")
					
					ner_label_list = label_list['ner']
					
					def compute_metrics(p):
						predictions, labels = p
						predictions = np.argmax(predictions, axis=2)
						

						# ner_label_list: ['B-PROT', 'B-SPECIES', 'I-PROT', 'I-SPECIES', 'O']


						# Remove ignored index (special tokens)
						true_predictions = [
							[ner_label_list[p] for (p, l) in zip(prediction, label) if l != -100]
							for prediction, label in zip(predictions, labels)
						]
						true_labels = [
							[ner_label_list[l] for (p, l) in zip(prediction, label) if l != -100]
							for prediction, label in zip(predictions, labels)
						]

						results = metric.compute(predictions=true_predictions, references=true_labels)
						'''
						if data_args.return_entity_level_metrics:
							# Unpack nested dictionaries
							final_results = {}
							for key, value in results.items():
								if isinstance(value, dict):
									for n, v in value.items():
										final_results[f"{key}_{n}"] = v
								else:
									final_results[key] = value
							return final_results
						else:
						'''
						return {
							"precision": results["overall_precision"],
							"recall": results["overall_recall"],
							"f1": results["overall_f1"],
							"accuracy": results["overall_accuracy"],
						}
					## [end] from NER


					result = compute_metrics((pred_output.predictions, pred_output.label_ids))
					
					with open(os.path.join(output_dir, data_name + '_' + learning_type + '_' + task_name + '_result.txt'), 'a') as fp:
						out_s = 'cv ' if do_cross_validation else ''
						out_s += "set: {set:d} / accuracy: {accuracy:.4f} / precision: {precision:.4f} / recall: {recall:.4f} / f1: {f1:.4f}\n".format(set=dataset_num, accuracy=result['accuracy'], precision=result['precision'], recall=result['recall'], f1=result['f1'])

						fp.write(out_s + '--------------------------\n')
						
					if do_cross_validation:
						all_results_per_data_and_task[data_name][task_name]['accuracy'].append(result['accuracy'])
						all_results_per_data_and_task[data_name][task_name]['precision'].append(result['precision'])
						all_results_per_data_and_task[data_name][task_name]['recall'].append(result['recall'])
						all_results_per_data_and_task[data_name][task_name]['f1'].append(result['f1'])

				elif task_name == 'ppi':
					
					'''
					print('preds_dict[task_name].predictions.shape:', preds_dict[task_name].predictions.shape)
					print('preds_dict[task_name].predictions[0]:', preds_dict[task_name].predictions[0])
					print('preds_dict[task_name].label_ids.shape:', preds_dict[task_name].label_ids.shape)
					print('preds_dict[task_name].label_ids[0]:', preds_dict[task_name].label_ids[0])
					'''
					
					t = torch.from_numpy(pred_output.predictions)
					
					pred = torch.softmax(t, dim=1).max(1)[1]
					true = pred_output.label_ids[:,0]
					
					
					if save_misclassified_samples:
						#miscls_indices = [pred.index(y) for x, y in zip(true, pred) if y != x]
						#miscls_sent = [datasets_dict[task_name][i]['input_ids'] for i in miscls_indices]
						#miscls_sent = [tokenizer_dict[task_name].decode(i) for i in miscls_sent]
	
						convert_id_to_str = ['enzyme', 'structural', 'negative'] # enzyme: 0, structural: 1, negative: 2
								
						for idx, elem in enumerate(true):
							if elem != pred[idx]:
								miscls_sent = datasets_dict[task_name][idx]['input_ids']
								miscls_sent = tokenizer_dict[task_name].decode(miscls_sent)
								miscls_sent = miscls_sent.replace('[CLS]', '').replace('[SEP]', '').strip()
								
								
								''' debug
								if miscls_sent in mis_sents:
									print('miscls_sent:', miscls_sent)
									input('enter..')
								else:
									mis_sents.append(miscls_sent)
								'''
								
								
								key = convert_id_to_str[elem] + '-' + convert_id_to_str[pred[idx]]
								if key in miscls_samples_per_data_and_task[data_name][task_name]:
									miscls_samples_per_data_and_task[data_name][task_name][key].append(miscls_sent)
								else:
									miscls_samples_per_data_and_task[data_name][task_name][key] = [miscls_sent]

					
					# debug
					#print('pred:', pred)
					#print('pred[0]:', pred[0])
					#print('true:', true)

					enzyme_cnt = 0
					structural_cnt = 0
					negative_cnt = 0
					
					for label in pred:
						if label == 0:
							enzyme_cnt += 1
						elif label == 1:
							structural_cnt += 1
						elif label == 2:
							negative_cnt += 1
					
					pred_stat = 'pred - enzyme_cnt: ' + str(enzyme_cnt) + ' / structural_cnt: ' + str(structural_cnt) + ' / negative_cnt: ' \
								   + str(negative_cnt) + ' / total: ' + str(enzyme_cnt + structural_cnt + negative_cnt)
					
					enzyme_cnt = 0
					structural_cnt = 0
					negative_cnt = 0
					
					for label in true:
						if label == 0:
							enzyme_cnt += 1
						elif label == 1:
							structural_cnt += 1
						elif label == 2:
							negative_cnt += 1
							
					true_stat = 'true - enzyme_cnt: ' + str(enzyme_cnt) + ' / structural_cnt: ' + str(structural_cnt) + ' / negative_cnt: ' \
								   + str(negative_cnt) + ' / total: ' + str(enzyme_cnt + structural_cnt + negative_cnt)
					
					from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

					# LBERT used micro scores. I also tested with micro scores, but it wan't quite different from weighted. 	
					precision, recall, f1, _ = precision_recall_fscore_support(true, pred, average='weighted')
					#precision, recall, f1, _ = precision_recall_fscore_support(true, pred, average='micro')
					
					result = {
						"accuracy": accuracy_score(true, pred),
						"precision": precision,
						"recall": recall,
						"f1": f1
					}
					cm = confusion_matrix(true, pred)
					
					logger.info("***** Eval results *****")
					for key in sorted(result.keys()):
						logger.info("  %s = %s", key, str(result[key]))
					
					with open(os.path.join(output_dir, data_name + '_' + learning_type + '_' + task_name + '_result.txt'), 'a') as fp:
						out_s = 'cv ' if do_cross_validation else ''
						out_s += "set: {set:d} / accuracy: {accuracy:.4f} / precision: {precision:.4f} / recall: {recall:.4f} / f1: {f1:.4f}\n".format(set=dataset_num, accuracy=result['accuracy'], precision=result['precision'], recall=result['recall'], f1=result['f1'])
						out_s += true_stat + '\n'
						out_s += pred_stat + '\n'
						out_s += str(cm.tolist()) + '\n'
						fp.write(out_s + '--------------------------\n')
					
					if do_cross_validation:
						all_results_per_data_and_task[data_name][task_name]['accuracy'].append(result['accuracy'])
						all_results_per_data_and_task[data_name][task_name]['precision'].append(result['precision'])
						all_results_per_data_and_task[data_name][task_name]['recall'].append(result['recall'])
						all_results_per_data_and_task[data_name][task_name]['f1'].append(result['f1'])
			
	if do_cross_validation:
		for data_name, tasks in all_results_per_data_and_task.items():
			for task_name, all_results in tasks.items():
				all_accuracy = np.array(all_results['accuracy'])
				all_precision = np.array(all_results['precision'])
				all_recall = np.array(all_results['recall'])
				all_f1 = np.array(all_results['f1'])
		
				with open(os.path.join(output_dir, data_name + '_' + learning_type + '_' + task_name + '_result.txt'), 'a') as fp:
					fp.write(">> Average Accuracy: %.4f with a std of %.4f\n" % (all_accuracy.mean(), all_accuracy.std()))
					fp.write(">> Average Precision: %.4f with a std of %.4f\n" % (all_precision.mean(), all_precision.std()))
					fp.write(">> Average Recall: %.4f with a std of %.4f\n" % (all_recall.mean(), all_recall.std()))
					fp.write(">> Average F1: %.4f with a std of %.4f\n" % (all_f1.mean(), all_f1.std()))
	
	if save_misclassified_samples:
		for data_name, tasks in miscls_samples_per_data_and_task.items():
			for task_name, mis_samples in tasks.items():
				for true_pred, sents in mis_samples.items():
					with open(os.path.join(output_dir, data_name + '_' + learning_type + '_' + task_name + '_miscls_samples_(' + true_pred + ').txt'), 'a') as fp:
						for sent in sents:
							fp.write('%s\n' % sent)


def main():
	
	## [start] from NER
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
	## [end] from NER	
	
	model_list = model_args.model_list
	task_list = data_args.task_list
	output_dir = training_args.output_dir
	
	do_fine_tune = data_args.do_fine_tune  # only for MTL
	
	if len(task_list) == 1 and do_fine_tune is True:
		sys.exit('ERROR: do_fine_tune is set for STL!!')

	for model_name in model_list:
	
		model_args.model_name_or_path = model_name
		training_args.output_dir = os.path.join(output_dir, model_name)
		
		## [start] from NER
		
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

		# Setup logging
		logging.basicConfig(
			format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
			datefmt="%m/%d/%Y %H:%M:%S",
			handlers=[logging.StreamHandler(sys.stdout)],
		)
		logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

		# Log on each process the small summary:
		logger.warning(
			f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
			+ f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
		)
		# Set the verbosity to info of the Transformers logger (on main process only):
		if is_main_process(training_args.local_rank):
			transformers.utils.logging.set_verbosity_info()
			transformers.utils.logging.enable_default_handler()
			transformers.utils.logging.enable_explicit_format()
		logger.info("Training/evaluation parameters %s", training_args)

		# Set seed before initializing model.
		set_seed(training_args.seed)

		## [end] from NER
		
		
		# debugging
		'''
		print('model_args:', model_args)
		print('data_args:', data_args)
		print('training_args:', training_args)
		sys.exit(1)
		'''

		num_of_datasets = 1 # if not CV, there is only one dataset.
		if data_args.do_cross_validation:
			num_of_datasets = data_args.num_of_folds

		results_dict = {} # store prediction results for each dataset and task.
		label_list = {} # label_list['ner'] is used when storing NER results.
		
		for dataset_num in range(num_of_datasets):
			logger.info("\n\n*** Dataset number: " + str(dataset_num) + " ***\n\n")


			'''
			# this is used to create the ner dataset.
			dataset_dict = {}
			dataset_dict['joint-ner-ppi'] = read_dataset(dataset_num, 'joint-ner-ppi', data_args)
			continue
			'''
			
			

			# Load Datasets.
			dataset_dict = {}
			for task_name in task_list:
				dataset_dict[task_name] = read_dataset(dataset_num, task_name, data_args)

				if task_name == 'ner' or task_name == 'joint-ner-ppi': # label_list['ner'] is used when storing NER results.
					if 'ner' not in label_list:
						if training_args.do_train:
							label_list['ner'] = get_label_list(dataset_dict[task_name]["train"]["ner"])
						else:
							label_list['ner'] = get_label_list(dataset_dict[task_name]["test"]["ner"])	
				
				# this is not used for now.
				'''
				if task_name == 'ppi':
					if 'ppi' not in label_list:
						label_list['ppi'] = data_args.ppi_classes
				'''

			# debugging
			'''
			for task_name, dataset in dataset_dict.items():
				print(task_name)
				print(type(dataset["train"]))
				print(dataset["train"])
				print(type(dataset["train"][0]))
				print(dataset["train"][0])
				print()
				input('enter..')
			'''

			# Set up tokenizers.
			tokenizer_dict = {}
			for task_name in task_list:
				tokenizer_dict[task_name] = AutoTokenizer.from_pretrained(
												model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
												cache_dir=model_args.cache_dir,
												use_fast=True,
												revision=model_args.model_revision,
												use_auth_token=True if model_args.use_auth_token else None,
												do_lower_case=model_args.do_lower_case
											)
				
				if task_name == 'ner':
					## [start] from NER
					# Tokenizer check: this script requires a fast tokenizer.
					if not isinstance(tokenizer_dict[task_name], PreTrainedTokenizerFast):
						raise ValueError(
							"This example script only works for models that have a fast tokenizer. Checkout the big table of models "
							"at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
							"requirement"
						)
					## [end] from NER
				
				## commmented this to avoid an error when a model is fine-tuned. 07-12-2021
				#if task_name == 'ppi' or task_name == 'joint-ner-ppi':
				# all representations use entity markers except for STANDARD_cls_token. 
				# STANDARD_mention_pooling needs entity markers to find entities in a sentence.
				if data_args.relation_representation != 'STANDARD_cls_token':
					if model_args.do_lower_case:
						tokenizer_dict[task_name].add_tokens(['[e1]', '[/e1]', '[e2]', '[/e2]'])
					else:
						tokenizer_dict[task_name].add_tokens(['[E1]', '[/E1]', '[E2]', '[/E2]'])

				#save_as_pickle("%s_tokenizer.pkl" % model_name, tokenizer_dict[task_name])
				#logger.info("Saved %s tokenizer at ./data/%s_tokenizer.pkl" %(model_name, model_name))
				
				#e1_id = tokenizer_dict[task_name].convert_tokens_to_ids('[E1]')
				#e2_id = tokenizer_dict[task_name].convert_tokens_to_ids('[E2]')
				#assert e1_id != e2_id != 1
			
				if isinstance(tokenizer_dict[task_name], BertTokenizerFast):
					tokenizer_dict[task_name].model_max_length = 512 # BioBERT default max length is too large. 512 is a default value of BERT tokenizer.

				# debugging
				'''
				print(tokenizer_dict[task_name])
				print(isinstance(tokenizer_dict[task_name], BertTokenizerFast))
				print('tokenizer_dict[task_name].model_max_length:', tokenizer_dict[task_name].model_max_length)
				print(tokenizer_dict[task_name].tokenize("[E1]BioBERT[/E1] sets [E2]do_lower_case[/E2] to True by default. Test IT."))
				input('enter..')
				'''

			# Set up models.
			model_type_dict = {}
			for task_name in task_list:
				model_type_dict[task_name] = AutoModelForTokenClassification
				'''
				if task_name == 'ner':
					model_type_dict[task_name] = AutoModelForTokenClassification
				elif task_name == 'ppi':
					model_type_dict[task_name] = AutoModelForTokenClassification
					#model_type_dict[task_name] = BertForTokenClassification_PPI
				'''

			model_config_dict = {}
			for task_name in task_list:
				model_config_dict[task_name] = AutoConfig.from_pretrained(
												model_args.config_name if model_args.config_name else model_args.model_name_or_path,
												num_labels=get_num_of_labels(task_name, dataset_dict, training_args, data_args),
												finetuning_task=task_name,
												cache_dir=model_args.cache_dir,
												revision=model_args.model_revision,
												#use_auth_token=True if model_args.use_auth_token else None,
											 )
		
			multitask_model = MultitaskModel.create(
				model_name=model_name,
				model_args=model_args,
				model_type_dict=model_type_dict,
				model_config_dict=model_config_dict,
				data_args=data_args,
				tokenizer_dict=tokenizer_dict,
			)
			
			for task_name in task_list:
				## commmented this to avoid an error when a model is fine-tuned. 07-12-2021
				#if task_name == 'ppi' or task_name == 'joint-ner-ppi':
				if data_args.relation_representation != 'STANDARD_cls_token':
					multitask_model.taskmodels_dict[task_name].resize_token_embeddings(len(tokenizer_dict[task_name]))
					
			# debugging
			# To confirm that all three task-models use the same encoder, we can check the data pointers of the respective encoders.
			# In this case, we'll check that the word embeddings in each model all point to the same memory location.
			'''
			print(multitask_model.encoder.embeddings.word_embeddings.weight.data_ptr())
			if model_name.startswith("bert") or model_name.startswith("dmis-lab"):
				for task_name in task_list:
					print(multitask_model.taskmodels_dict[task_name].bert.embeddings.word_embeddings.weight.data_ptr())
			elif model_name.startswith("roberta-"):
				for task_name in task_list:
					print(multitask_model.taskmodels_dict[task_name].roberta.embeddings.word_embeddings.weight.data_ptr())
			else:
				print("Exercise for the reader: add a check for other model architectures =)")
			'''

			## [start] from NER
			# Preprocessing the dataset
			# Padding strategy
			padding = "max_length" if data_args.pad_to_max_length else False
			## [end] from NER

			features_dict = featurize_data(dataset_dict, tokenizer_dict, padding, data_args, model_args.do_lower_case)

			train_dataset = {task_name: dataset["train"] for task_name, dataset in features_dict.items()}
			eval_dataset = {task_name: dataset["validation"] for task_name, dataset in features_dict.items() if "validation" in dataset}
			test_dataset = {task_name: dataset["test"] for task_name, dataset in features_dict.items()}
			
			# debugging
			'''
			print('len(train_dataset[ppi]):', len(train_dataset['ppi']))
			print('type(train_dataset[ppi]):', type(train_dataset['ppi']))
			print('type(train_dataset[ppi][0]):', type(train_dataset['ppi'][0]))
			print(train_dataset['ppi'][0])
			print(tokenizer_dict['ppi'].decode(train_dataset['ppi'][0]['input_ids']))
			print(tokenizer_dict['ppi'].decode(train_dataset['ppi'][0]['input_ids']).replace('[CLS]', ''))
			input('enter..')
			'''
			
			# Data collator
			# TODO: assign a separate data collator for different tasks!
			if 'ppi' in task_list:
				data_collator = DataCollatorForTokenClassification(tokenizer_dict['ppi'], pad_to_multiple_of=8 if training_args.fp16 else None)
			else:
				data_collator = DataCollatorForTokenClassification(list(tokenizer_dict.values())[0], pad_to_multiple_of=8 if training_args.fp16 else None)
			
			
			
			
			
			
			'''
			# Metrics
			metric = load_metric("seqeval")
			
			ner_label_list = label_list['ner']
	
			def compute_metrics(p):
				predictions, labels = p
				predictions = np.argmax(predictions, axis=2)
				

				# ner_label_list: ['B-PROT', 'B-SPECIES', 'I-PROT', 'I-SPECIES', 'O']


				# Remove ignored index (special tokens)
				true_predictions = [
					[ner_label_list[p] for (p, l) in zip(prediction, label) if l != -100]
					for prediction, label in zip(predictions, labels)
				]
				true_labels = [
					[ner_label_list[l] for (p, l) in zip(prediction, label) if l != -100]
					for prediction, label in zip(predictions, labels)
				]

				results = metric.compute(predictions=true_predictions, references=true_labels)

				with open(os.path.join(output_dir, str(dataset_num) + '_mtl_' + task_name + '_result.txt'), 'a') as fp:
					fp.write(str(results))
					#for k, v in results.items():
					#	fp.write(k + ': ' + v + '\n')
					fp.write('\n')
			
			
				return {
					"precision": results["overall_precision"],
					"recall": results["overall_recall"],
					"f1": results["overall_f1"],
					"accuracy": results["overall_accuracy"],
				}
			'''
			
			
			

			
			trainer = MultitaskTrainer(
				model=multitask_model,
				#args=TrainingArguments(
				#	output_dir="./models/multitask_model",
				#	overwrite_output_dir=True,
				#	learning_rate=1e-5,	# different from default=5e-05
				#	do_train=True,
				#	num_train_epochs=3,
				#	# Adjust batch size if this doesn't fit on the Colab GPU
				#	per_device_train_batch_size=8,  
				#	save_steps=3000, # different from default=500
				#),
				args=training_args,
				#data_collator=NLPDataCollator(),
				data_collator=data_collator,
				train_dataset=train_dataset if training_args.do_train else None,
				#eval_dataset=eval_dataset if training_args.do_eval else None,
				#compute_metrics=compute_metrics, # TODO: this is a metric for NER not for PPI. So, remove it and evaluate validation set later as in the original code.
			)

			## [start] from NER
			''' reference
			# Initialize our Trainer
			trainer = Trainer(
				model=model,
				args=training_args,
				train_dataset=train_dataset if training_args.do_train else None,
				eval_dataset=eval_dataset if training_args.do_eval else None,
				tokenizer=tokenizer,
				data_collator=data_collator,
				compute_metrics=compute_metrics,
			)
			'''
			## [end] from NER

			trainer.train()

			if do_fine_tune:
				for task_name, task_model in trainer.model.taskmodels_dict.items(): # save a task-specific model for fine-tuning
					task_model.save_pretrained(os.path.join(training_args.output_dir, task_name))

			results_dict[dataset_num] = []
			
			
			
			"""
			# Evaluation
			preds_dict = {}
			datasets_dict = {} # used for save_misclassified_samples
			if training_args.do_eval:
				logger.info("*** Evaluate ***")
				
				for task_name in task_list:
										
					if "validation" not in features_dict[task_name]:
						continue
					
					# TODO: this needs to be fixed and made cleaner later.
					if task_name == 'joint-ner-ppi':
						trainer.model.taskmodels_dict[task_name].finetuning_task = 'ppi'

					eval_dataloader = DataLoaderWithTaskname(
										task_name,
										trainer.get_eval_dataloader(eval_dataset=features_dict[task_name]["validation"])
									  )
					#print(eval_dataloader.data_loader.collate_fn)
					
					# TODO: this needs to be fixed and made cleaner later.
					if task_name == 'joint-ner-ppi':
						preds_dict['ppi'] = trainer.prediction_loop(
												eval_dataloader, 
												description=f"Validation: {task_name}",
											)
					else:
						preds_dict[task_name] = trainer.prediction_loop(
													eval_dataloader, 
													description=f"Validation: {task_name}",
												)
					
					datasets_dict[task_name] = eval_dataloader.dataset
					
					'''
					t = torch.from_numpy(preds_dict[task_name].predictions)
					
					pred = torch.softmax(t, dim=1).max(1)[1]
					true = preds_dict[task_name].label_ids[:,0]
					
					print('len(pred):', len(pred))
					print('len(true):', len(true))
					print('len(eval_dataloader.dataset):', len(eval_dataloader.dataset))

					print(type(eval_dataloader.dataset))
					
					print(eval_dataloader.dataset[1])
					
					res_list = [eval_dataloader.dataset[i] for i in [1, 2, 4]]
					
					print(res_list)
					
					for idx, data in enumerate(eval_dataloader.dataset):
						print('idx:', idx)
						print('pred[idx]:', pred[idx])
						print('true[idx]:', true[idx])
						print(data)
						input('enter..')
					'''							
											
				results_dict[dataset_num].append({'data_name': 'eval', 'preds_dict': preds_dict, 'datasets_dict': datasets_dict, 'tokenizer_dict': tokenizer_dict})
			"""
			
			# Prediction
			preds_dict = {}
			datasets_dict = {}
			if training_args.do_predict:
				logger.info("*** Predict ***")
				
				for task_name in task_list:
					
					if do_fine_tune:
						task_model = AutoModelForTokenClassification.from_pretrained(os.path.join(training_args.output_dir, task_name),
																					 relation_representation=data_args.relation_representation,
																					 num_ppi_labels=len(data_args.ppi_classes),
																					 tokenizer=tokenizer_dict[task_name])

						task_train_dataset = train_dataset[task_name]
						
						task_trainer = Trainer(
							model=task_model,
							args=training_args,
							data_collator=data_collator,
							train_dataset=task_train_dataset
						)
						
						#print(task_trainer.finetuning_task)
						#input('enter..')

						task_trainer.train()
						test_dataloader = trainer.get_test_dataloader(test_dataset=features_dict[task_name]["test"])	
					else:
						test_dataloader = DataLoaderWithTaskname(
											task_name,
											trainer.get_test_dataloader(test_dataset=features_dict[task_name]["test"])
										  )
						#print(test_dataloader.data_loader.collate_fn)
					
					# TODO: this needs to be fixed and made cleaner later.
					if task_name == 'joint-ner-ppi':
 
						trainer.model.taskmodels_dict[task_name].finetuning_task = 'ner'
						
						preds_dict['ner'] = trainer.prediction_loop(
												test_dataloader, 
												description=f"Test: {task_name}",
											)
						
						test_dataloader = DataLoaderWithTaskname(
											task_name,
											trainer.get_test_dataloader(test_dataset=features_dict[task_name]["validation"])
										  )
									  
						trainer.model.taskmodels_dict[task_name].finetuning_task = 'ppi'

						preds_dict['ppi'] = trainer.prediction_loop(
												test_dataloader, 
												description=f"Test: {task_name}",
											)
					else:
						if do_fine_tune:
							preds_dict[task_name] = task_trainer.prediction_loop(
														test_dataloader, 
														description=f"Test: {task_name}",
													)
						else:
							preds_dict[task_name] = trainer.prediction_loop(
													test_dataloader, 
													description=f"Test: {task_name}",
												)
						
					datasets_dict[task_name] = test_dataloader.dataset
			
				results_dict[dataset_num].append({'data_name': 'pred', 'preds_dict': preds_dict, 'datasets_dict': datasets_dict, 'tokenizer_dict': tokenizer_dict})
			
			if do_fine_tune:
				for task_name in task_list:  # delete saved models.
					model_dir = os.path.join(training_args.output_dir, task_name)
					try:
						shutil.rmtree(model_dir)
					except OSError as e:
						print("Error: %s : %s" % (model_dir, e.strerror))

		logger.info("*** Data iterations are done.  ***")
		
		if len(results_dict[0]) > 0:
			data_list = []
			if training_args.do_eval:
				data_list.append('eval')

			if training_args.do_predict:
				data_list.append('pred')

			# TODO: this needs to be fixed and made cleaner later.
			if task_name == 'joint-ner-ppi':
				task_list = ['ner', 'ppi']
				
			save_results(results_dict, data_list, task_list, label_list, \
						 training_args.output_dir, \
						 data_args.do_cross_validation, data_args.save_misclassified_samples)
		
		# TODO: this needs to be fixed and made cleaner later.
		if task_name == 'joint-ner-ppi':
			task_list = data_args.task_list


if __name__ == "__main__":
    main()