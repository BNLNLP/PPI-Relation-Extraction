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

### TODO: remove this if not used.
# this is used for DDI evaluation. Remove it once DDI evaluation is combined with seqeval.
from sklearn.metrics import f1_score, accuracy_score, classification_report, recall_score, precision_score, precision_recall_fscore_support

### TODO: uncomment this after test.
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
#check_min_version("4.17.0")

### TODO: define requirements by referring the following examples.
#require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/token-classification/requirements.txt")
#require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")


dataset_list = ["ChemProt_BLURB", "DDI_BLURB", "GAD_BLURB", "EU-ADR_BioBERT",
                "AImed", "BioInfer", "HPRD50", "IEPA", "LLL",
                "Typed_PPI",
                "P-putida", "P-species"]

dataset_max_seq_length = {
    #"ChemProt_BLURB": 256, # some samples (count: 11) are longer than 256 tokens.
    #"DDI_BLURB": 256, # many samples are longer than 256 tokens. 
    #"GAD_BLURB": 128, # some samples (count: 3) are longer than 128 tokens when BioBERT is used.
    #"EU-ADR_BioBERT": 128, # some samples (count: 14) are longer than 128 tokens.
}

dataset_special_tokens = {
    "ChemProt_BLURB": ["@GENE$", "@CHEMICAL$", "@CHEM-GENE$"],
    "DDI_BLURB": ["@DRUG$", "@DRUG-DRUG$"],
    "GAD_BLURB": ["@GENE$", "@DISEASE$"],
    "EU-ADR_BioBERT": ["@GENE$", "@DISEASE$"],
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
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
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
        default="STANDARD_mention_pooling",
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
            d_name = self.dataset_name.rsplit('/', 1)[1] if len(self.dataset_name.rsplit('/', 1)) > 1 else self.dataset_name
            if d_name not in dataset_list:
                raise ValueError("Unknown dataset, you should pick one in " + ", ".join(dataset_list) + ". Or, add your dataset to the dataset list.")
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


#
def save_results(
    results_dict, 
    data_list, 
    task_name, 
    label_list, 
    output_dir,
    do_cross_validation, 
    save_misclassified_samples,
):
    
    if do_cross_validation:
        all_results_per_data_and_task = {}
        for data_name in data_list: # eval or pred
            all_results_per_data_and_task[data_name][task_name] = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    if save_misclassified_samples:
        miscls_samples_per_data_and_task = {}
        for data_name in data_list: # eval or pred
            miscls_samples_per_data_and_task[data_name][task_name] = {}
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for dataset_num, results in results_dict.items():
        for result in results:
            data_name = result['data_name'] # eval or pred
            prediction_output = result['prediction_output']
            datasets = result['datasets']
            tokenizer = result['tokenizer']						  

            predictions, labels = prediction_output.predictions, prediction_output.label_ids
            
            '''
            print(predictions)
            print(labels)
            print(predictions.shape)
            print(labels.shape)
            '''
            
            #predictions = np.argmax(predictions, axis=2)
            predictions = np.argmax(predictions, axis=1)
            labels = labels.flatten()
            
            '''
            print(predictions)
            print(labels)
            print(predictions.shape)
            print(labels.shape)
            #input('enter..')
            '''
            
            pred = [label_list[x] for x in predictions]
            true = [label_list[x] for x in labels]
            
            
            # Remove ignored index (special tokens)
            '''
            true_predictions = [
                [task_label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            true_labels = [
                [task_label_list[l] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            '''
            
            # debug
            '''
            print(task_name, pred_output.predictions)
            print(task_name, pred_output.predictions.shape)
            print(task_name, pred_output.predictions[:, :true_label_len, :])
            print(task_name, pred_output.predictions[:, :true_label_len, :].shape)
            print(task_name, pred_output.label_ids)
            print(task_name, pred_output.label_ids.shape)
            input('enter..')
            '''

            
            if 'DDI-false' in label_list: # for DDI, ignore 'DDI-false' label.
                cleaned_pred_labels = []
                for i in pred:
                    if i == 'DDI-false':
                        cleaned_pred_labels.append('false')
                    else:
                        cleaned_pred_labels.append(i)
                pred = cleaned_pred_labels
                label_list.remove('DDI-false')
            elif 'no_relation' in label_list: # for TACRED, ignore 'no_relation' label.
                cleaned_pred_labels = []
                for i in pred:
                    if i == 'no_relation':
                        cleaned_pred_labels.append('false')
                    else:
                        cleaned_pred_labels.append(i)
                pred = cleaned_pred_labels
                label_list.remove('no_relation')

            # debug
            '''
            with open(os.path.join(output_dir, data_name + '_' + task_name + '_pred_labels.txt'), 'a') as fp:
                for element in predictions:
                    fp.write(element + "\n")
            with open(os.path.join(output_dir, data_name + '_' + task_name + '_actual_labels.txt'), 'a') as fp:
                for element in true:
                    fp.write(element + "\n")
            '''

            #f1score = f1_score(y_pred=pred, y_true=true, average='micro', labels=label_list)
            precision, recall, f1, _ = precision_recall_fscore_support(y_pred=pred, y_true=true, \
                                                                       labels=label_list, average='micro')
            accuracy = accuracy_score(true, pred) # TODO: ignore 'DDI-false' for DDI evaluation.
            
            print("<sklearn> - classification_report")
            print(classification_report(true, pred, digits=4, labels=label_list))
            print('<sklearn> precision:', precision)
            print('<sklearn> recall:', recall)
            print('<sklearn> f1:', f1) # this is the same as f1_score.
            print('<sklearn> accuracy:', accuracy)

            result = {"precision": precision, 
                      "recall": recall,
                      "f1": f1,
                      "accuracy": accuracy}

            with open(os.path.join(output_dir, data_name + '_' + task_name + '_result.txt'), 'a') as fp:
                out_s = 'cv ' if do_cross_validation else ''
                out_s += "set: {set:d} / accuracy: {accuracy:.4f} / precision: {precision:.4f} / recall: {recall:.4f} / f1: {f1:.4f}\n".format(set=dataset_num, accuracy=result['accuracy'], precision=result['precision'], recall=result['recall'], f1=result['f1'])

                fp.write(out_s + '--------------------------\n')
                
            if do_cross_validation:
                all_results_per_data_and_task[data_name][task_name]['accuracy'].append(result['accuracy'])
                all_results_per_data_and_task[data_name][task_name]['precision'].append(result['precision'])
                all_results_per_data_and_task[data_name][task_name]['recall'].append(result['recall'])
                all_results_per_data_and_task[data_name][task_name]['f1'].append(result['f1'])
            
                
            
            ### TODO: fix this!!
            if save_misclassified_samples:
                #miscls_indices = [pred.index(y) for x, y in zip(true, pred) if y != x]
                #miscls_sent = [datasets_dict[task_name][i]['input_ids'] for i in miscls_indices]
                #miscls_sent = [tokenizer_dict[task_name].decode(i) for i in miscls_sent]

                #print('len(pred):', len(pred))
                #print('len(true):', len(true))
                

                for idx, (p_labels, t_labels) in enumerate(zip(pred, true)):
                    input_ids = datasets_dict[task_name][idx]['input_ids']
                    miscls_sent = tokenizer_dict[task_name].decode(input_ids)
                    miscls_sent = miscls_sent.replace('[CLS]', '').replace('[SEP]', '').strip()
                    
                    def divide_chunks(l, n):
                        for i in range(0, len(l), n): 
                            yield l[i:i + n]
                            
                    rels = list(divide_chunks(datasets_dict[task_name][idx]['relations'], 5))
                    
                    
                    '''
                    #for tt in datasets_dict[task_name][idx]:
                    #	print(tt)

                    print(len(p_labels))
                    print(len(t_labels))
                    print(len(rels))
                    print(p_labels)
                    print(t_labels)
                    print(rels)
                    print(datasets_dict[task_name][idx]['input_ids'])
                    tt = datasets_dict[task_name][idx]['input_ids']
                    print(tokenizer_dict[task_name].decode(tt))
                    '''
                    
                    #input('enter..')
                    
                    
                    
                    
                    ''' debug
                    if miscls_sent in mis_sents:
                        print('miscls_sent:', miscls_sent)
                        input('enter..')
                    else:
                        mis_sents.append(miscls_sent)
                    '''
                    # debug
                    if len(p_labels) != len(t_labels):
                        print('Error - difference in label length!!')
                        input('enter..')
                    
                    for p_label, t_label, rel in zip(p_labels, t_labels, rels):
                        if p_label != t_label:
                            key = t_label + '-' + p_label
                            #key = p_label
                            
                            # relation_spans -> [e1_span_s, e1_span_e, e2_span_s, e2_span_e, rel['rel_id']]
                            e1 = input_ids[rel[0]:rel[1]]
                            e2 = input_ids[rel[2]:rel[3]]
                            e1 = tokenizer_dict[task_name].decode(e1)
                            e2 = tokenizer_dict[task_name].decode(e2)

                            e_and_miscls_sent = e1 + ' - ' + e2 + ' | ' + miscls_sent
                            
                            if key in miscls_samples_per_data_and_task[data_name][task_name]:
                                miscls_samples_per_data_and_task[data_name][task_name][key].append(e_and_miscls_sent)
                            else:
                                miscls_samples_per_data_and_task[data_name][task_name][key] = [e_and_miscls_sent]

            
                
    if do_cross_validation:
        for data_name, tasks in all_results_per_data_and_task.items():
            for task_name, all_results in tasks.items():
                all_accuracy = np.array(all_results['accuracy'])
                all_precision = np.array(all_results['precision'])
                all_recall = np.array(all_results['recall'])
                all_f1 = np.array(all_results['f1'])
        
                with open(os.path.join(output_dir, data_name + '_' + task_name + '_result.txt'), 'a') as fp:
                    fp.write(">> Average Accuracy: %.4f with a std of %.4f\n" % (all_accuracy.mean(), all_accuracy.std()))
                    fp.write(">> Average Precision: %.4f with a std of %.4f\n" % (all_precision.mean(), all_precision.std()))
                    fp.write(">> Average Recall: %.4f with a std of %.4f\n" % (all_recall.mean(), all_recall.std()))
                    fp.write(">> Average F1: %.4f with a std of %.4f\n" % (all_f1.mean(), all_f1.std()))
    
    ### TODO: fix this!!
    if save_misclassified_samples:
        for data_name, tasks in miscls_samples_per_data_and_task.items():
            for task_name, mis_samples in tasks.items():
                for true_pred, sents in mis_samples.items():
                    with open(os.path.join(output_dir, data_name + '_' + task_name + '_miscls_samples_(' + true_pred + ').txt'), 'a') as fp:
                        for sent in sents:
                            fp.write('%s\n' % sent)


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
    
    # Load relation and entity types.
    relation_type_file = os.path.join(data_args.dataset_dir, dataset_name)
    relation_type_file = os.path.join(relation_type_file, "relation_types.json")
    relation_types = json.load(open(relation_type_file))
    
    entity_type_file = os.path.join(data_args.dataset_dir, dataset_name)
    entity_type_file = os.path.join(entity_type_file, "entity_types.json")
    entity_types = json.load(open(entity_type_file))	
    
    label_list = list(relation_types.keys())
        
    initial_output_dir = training_args.output_dir
    
    if not os.path.exists(initial_output_dir):
        os.makedirs(initial_output_dir)
    
    for model_name in model_list:
        model_args.model_name_or_path = model_name
        training_args.output_dir = initial_output_dir
        
        training_args.output_dir = os.path.join(training_args.output_dir, dataset_name)
        training_args.output_dir = os.path.join(training_args.output_dir, model_name)
        training_args.output_dir = os.path.join(training_args.output_dir, relation_representation)
        
        if data_args.use_context != None:
            if data_args.use_context == "attn_based":
                training_args.output_dir += "_ac"
            elif data_args.use_context == "local":
                training_args.output_dir += "_lc"
        
        if data_args.use_entity_type_embeddings:
            training_args.output_dir += "_et"
            
        if data_args.use_entity_typed_marker:
            training_args.output_dir += "_tm"

        # Set seed before initializing model.
        set_seed(training_args.seed)

        # Padding strategy
        if data_args.pad_to_max_length:
            padding = "max_length"
        else:
            # We will pad later in data collator, dynamically at batch creation, to the max sequence length in each batch
            padding = False
            
        '''
        def get_model():

            #print(config.vocab_size)
            
            config = AutoConfig.from_pretrained(
                model_args.config_name if model_args.config_name else model_args.model_name_or_path,
                num_labels=len(label_list),
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
        '''

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
            
            if any(x == dataset_name for x in ["GAD_BLURB", "EU-ADR_BioBERT"]):
                average = "binary"
            else:
                average = "micro"
            
            a = a_m.compute(predictions=pred, references=true)
            p = p_m.compute(predictions=pred, references=true, average=average)
            r = r_m.compute(predictions=pred, references=true, average=average)
            f = f_m.compute(predictions=pred, references=true, average=average)
            
            return {"accuracy": a["accuracy"], "precision": p["precision"], "recall": r["recall"], "f1": f["f1"]}
        
        ### TODO: complete this.
        results_dict = {} # store prediction results for each dataset and task.
        
        # Remove old output files except the cross-validation result file.
        if os.path.exists(training_args.output_dir):
            for f in os.listdir(training_args.output_dir):
                if os.path.isfile(os.path.join(training_args.output_dir, f)) and f != "predict_results_history.json":
                    os.remove(os.path.join(training_args.output_dir, f))
            
        # Get the number of datasets. If it's more than 1, it's a cross-validation dataset.
        num_of_datasets = 0
        for file in os.listdir(os.path.join(data_args.dataset_dir, data_args.dataset_name)):
            if file.startswith('train_') and file.endswith('.json'):
                num_of_datasets += 1

        for dataset_num in range(num_of_datasets):
            logger.info("\n\n*** Dataset number: " + str(dataset_num) + " ***\n\n")

            # Load pretrained model and tokenizer
            #
            # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
            # download model & vocab.
            config = AutoConfig.from_pretrained(
                model_args.config_name if model_args.config_name else model_args.model_name_or_path,
                num_labels=len(label_list),
                finetuning_task=task_name,
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
            
            # Remove any parent directories of the dataset if exist.
            dataset_name = dataset_name.rsplit('/', 1)[1] if len(dataset_name.rsplit('/', 1)) > 1 else dataset_name
            
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
            additional_special_tokens = []
            if relation_representation.startswith('EM'):
                #additional_special_tokens = {"additional_special_tokens": entity_marker_special_tokens[dataset_name] if use_entity_typed_marker \
                #                                                          else entity_marker_special_tokens['EM']}
                additional_special_tokens.extend(entity_marker_special_tokens[dataset_name] if use_entity_typed_marker \
                                                                                            else entity_marker_special_tokens['EM'])                                                              
                #tokenizer.add_special_tokens(additional_special_tokens)
            
            # Add the special tokens that are used to replace entity names (entity anonymization or dummification). E.g,. ChemProt, DDI, GAD
            if dataset_name in dataset_special_tokens.keys():
                #additional_special_tokens = {"additional_special_tokens": dataset_special_tokens[dataset_name]}
                additional_special_tokens.extend(dataset_special_tokens[dataset_name])
                
                #tokenizer.add_special_tokens(additional_special_tokens)
            
            # Add additional special tokens at once. If they are added separately, then only tokens added later remain. 05/04/2022
            if len(additional_special_tokens) > 0:
                tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
            
            
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
            
            print(tokenizer.all_special_ids)
            print(tokenizer.all_special_tokens) 
            print(tokenizer.all_special_tokens_extended)
            print(tokenizer.additional_special_tokens_ids)
            print(tokenizer.get_added_vocab)
            
            print(tokenizer)
            #print(isinstance(tokenizer, BertTokenizerFast))
            print('tokenizer.model_max_length:', tokenizer.model_max_length)
            print(tokenizer.tokenize("[E1]BioBERT[/E1] sets [E2]do_lower_case[/E2] to True by default. Test IT."))
            input('enter..')
            '''
            
            
            
            
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
                )
                
            # Resize input token embeddings matrix of the model since new tokens have been added.
            # this funct is used if the number of tokens in tokenizer is different from config.vocab_size.
            model.resize_token_embeddings(len(tokenizer))


            # Loading a dataset from your local files.
            data_files = read_dataset(dataset_num, task_name, data_args)

            # debugging
            '''
            print(type(data_files["train"]))
            print(data_files["train"])
            print(type(data_files["train"][0]))
            print(data_files["train"][0])
            print()
            input('enter..')
            '''
            
            # This is a temporary code.
            # In BioInfer, some entities consist of separate tokens in a text, and the partial tokens are not properly tokenized by tokenizer.
            # To avoid the mismatch between entity index from data file and entity index from tokenized output, add the partial tokens to the tokenzier
            # so that the tokenizer properly catch partial tokens.
            # E.g., "GP IIIa" in "GPIIb-IIIa", "MEK 2" in "MEK1/2"
            if dataset_name == "BioInfer":
                partial_token_list = []
                for d in concatenate_datasets([data_files["train"], data_files["test"]]):
                    if len(d['relation'][0]['entity_1_idx']) > 1:
                        for (s, e) in d['relation'][0]['entity_1_idx']:
                            if d['text'][s-1] != ' ' or d['text'][e] != ' ':
                                partial_token = d['text'][s:e]
                                partial_token_list.extend(partial_token.split())

                partial_token_list = list(set(partial_token_list))
                tokenizer.add_tokens(partial_token_list)

                # Resize input token embeddings matrix of the model since new tokens have been added.
                # this funct is used if the number of tokens in tokenizer is different from config.vocab_size.
                model.resize_token_embeddings(len(tokenizer))


            dataset = featurize_data(data_files, tokenizer, padding, max_seq_length, relation_representation, use_entity_typed_marker)

            train_dataset = dataset["train"]
            eval_dataset = dataset["validation"] if "validation" in dataset else None
            test_dataset = dataset["test"]
            
            # debugging
            '''
            for x in train_dataset:
                print(x)
                input('enter..')
            
            print('len(train_dataset[ppi]):', len(train_dataset['ppi']))
            print('type(train_dataset[ppi]):', type(train_dataset['ppi']))
            print('type(train_dataset[ppi][0]):', type(train_dataset['ppi'][0]))
            print(train_dataset['ppi'][0])
            print(tokenizer_dict['ppi'].decode(train_dataset['ppi'][0]['input_ids']))
            print(tokenizer_dict['ppi'].decode(train_dataset['ppi'][0]['input_ids']).replace('[CLS]', ''))
            input('enter..')
            '''
            
            # Data collator
            data_collator = DataCollatorForRelationClassification(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
            
            # Initialize our Trainer
            trainer = Trainer(
                model=model,
                #model_init=get_model,
                args=training_args,
                train_dataset=train_dataset if training_args.do_train else None,
                eval_dataset=eval_dataset if training_args.do_eval else None,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            )
                    
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
            
            # Training
            if training_args.do_train:
                checkpoint = None
                if training_args.resume_from_checkpoint is not None:
                    checkpoint = training_args.resume_from_checkpoint
                elif last_checkpoint is not None:
                    checkpoint = last_checkpoint
                train_result = trainer.train(resume_from_checkpoint=checkpoint)
                metrics = train_result.metrics
                max_train_samples = (
                    data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
                )
                metrics["train_samples"] = min(max_train_samples, len(train_dataset))

                #trainer.save_model()  # Saves the tokenizer too for easy upload

                trainer.log_metrics("train", metrics)
                trainer.save_metrics("train", metrics)
                trainer.save_state()
            
            ### TODO: complete this.
            results_dict[dataset_num] = []
            preds_dict = {}
            datasets_dict = {} # used for save_misclassified_samples
            
            eval_info = {"dataset_num": dataset_num, 
                         "seed": training_args.seed,
                         "epoch": training_args.num_train_epochs,
                         "per_device_train_batch_size": training_args.per_device_train_batch_size,
                         "n_gpu": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                         "learning_rate": training_args.learning_rate,
                         "warmup_ratio": training_args.warmup_ratio,
                         "weight_decay": training_args.weight_decay}

            # Evaluation
            if training_args.do_eval:
                logger.info("*** Evaluate ***")

                metrics = trainer.evaluate(eval_dataset=eval_dataset)

                max_eval_samples = (
                    data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
                )
                metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

                trainer.log_metrics("eval", metrics)
                trainer.save_metrics("eval", metrics, combined=False, eval_info=eval_info)

            # Prediction
            if training_args.do_predict:
                logger.info("*** Predict ***")

                predictions, labels, metrics = trainer.predict(test_dataset, metric_key_prefix="predict")
                
                trainer.log_metrics("predict", metrics)
                trainer.save_metrics("predict", metrics, eval_info=eval_info)
                
                if data_args.save_predictions:
                    output_predict_file = os.path.join(training_args.output_dir, f"predict_outputs.txt")
                    if trainer.is_world_process_zero():
                        predictions = np.argmax(predictions, axis=1)
                        with open(output_predict_file, "a") as writer:
                            logger.info(f"***** Predict outputs *****")
                            if os.path.getsize(output_predict_file) == 0:
                                writer.write("index\tentity_1\tentity_2\ttext\tprediction\tlabel\n")
                            for index, (sample, item, label) in enumerate(zip(test_dataset, predictions, labels)):
                                item = label_list[item]
                                
                                # debug
                                if sample['labels'] != label:
                                    raise Exception("Label mismatch!!")
                                
                                ### TODO: for now, each sample has a single relation. Handle multiple relations in a sentence later.
                                label = label_list[label[0]]

                                input_ids = sample['input_ids']
                                sent = tokenizer.decode(input_ids)
                                sent = sent.replace('[CLS]', '').replace('[SEP]', '').strip()

                                def divide_chunks(l, n):
                                    for i in range(0, len(l), n): 
                                        yield l[i:i+n]
                                
                                ### TODO: for now, each sample has a single relation. Handle multiple relations in a sentence later.
                                e1_span_idx_list, e2_span_idx_list = list(divide_chunks(sample['relations'], 2))[0]

                                e1 = ' '.join([tokenizer.decode(input_ids[s:e]) for s, e in e1_span_idx_list])
                                e2 = ' '.join([tokenizer.decode(input_ids[s:e]) for s, e in e2_span_idx_list])

                                writer.write(f"{index}\t{e1}\t{e2}\t{sent}\t{item}\t{label}\n")
                                


            '''
            # Prediction
            if training_args.do_predict:
                logger.info("*** Predict ***")
                
                test_dataloader = trainer.get_test_dataloader(test_dataset=test_dataset)
                
                prediction_output = trainer.prediction_loop(
                    test_dataloader, 
                    description=f"Test: {task_name}",
                )
                
                # 'datasets' is used for save_misclassified_samples
                results_dict[dataset_num].append({'data_name': 'pred', 
                                                  'prediction_output': prediction_output, 
                                                  'datasets': test_dataloader.dataset,
                                                  'tokenizer': tokenizer})
            '''
            
            # Delete the saved model.
            '''
            model_dir = os.path.join(training_args.output_dir, task_name)
            try:
                shutil.rmtree(model_dir)
            except OSError as e:
                print("Error: %s : %s" % (model_dir, e.strerror))
            '''
        
        if training_args.do_predict:
            # Save the prediction results in a history file.
            result_file = os.path.join(training_args.output_dir, 'predict_results.json')
            f = open(result_file)
            
            txt = ''
            for line in f.readlines():
                line = line.strip()
                if line == '}{':
                    line = '}\n{'
                txt += line
            
            txt = txt.split('\n')
            
            entries = [json.loads(x) for x in txt]
            
            result_data = {}
            for entry in entries:
                for k, v in entry.items():
                    if k in result_data:
                        result_data[k].append(v)
                    else:
                        result_data[k] = [v]
            
            out_data = {}
            
            # Add a predix for average scores of CV datasets.
            key_prefix = "avg_" if num_of_datasets > 1 else ""
            
            out_data[key_prefix + "predict_f1"] = float(np.mean(result_data["predict_f1"]))
            out_data[key_prefix + "predict_precision"] = float(np.mean(result_data["predict_precision"]))
            out_data[key_prefix + "predict_recall"] = float(np.mean(result_data["predict_recall"]))
            out_data[key_prefix + "predict_accuracy"] = float(np.mean(result_data["predict_accuracy"]))
            out_data[key_prefix + "predict_loss"] = float(np.mean(result_data["predict_loss"]))
            out_data["epoch"] = result_data["epoch"][0]
            out_data["per_device_train_batch_size"] = result_data["per_device_train_batch_size"][0]
            out_data["learning_rate"] = str(result_data["learning_rate"][0])
            out_data["warmup_ratio"] = result_data["warmup_ratio"][0]
            out_data["weight_decay"] = result_data["weight_decay"][0]
            out_data["n_gpu"] = result_data["n_gpu"][0]
            out_data["num_of_datasets"] = num_of_datasets
            
            if num_of_datasets > 1:
                trainer.log_metrics(f"{num_of_datasets} folds cross-validation predict", out_data)
            
            out_file = os.path.join(training_args.output_dir, 'predict_results_history.json')
            with open(out_file, "a") as f:
                json.dump(out_data, f, indent=4, sort_keys=True)
            
        logger.info("*** Data iterations are done.  ***")
        
        '''
        if len(results_dict[0]) > 0:
            data_list = []
            if training_args.do_eval:
                data_list.append('eval')

            if training_args.do_predict:
                data_list.append('pred')

            save_results(results_dict,
                         data_list,
                         task_name,
                         label_list,
                         training_args.output_dir,
                         data_args.do_cross_validation, 
                         data_args.save_misclassified_samples)
        '''

if __name__ == "__main__":
    main()
    
    