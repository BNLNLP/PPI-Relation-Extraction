import os
import sys
import re
import pickle
import json
import pandas as pd
import numpy as np
from datasets import ClassLabel, load_dataset, load_metric, Dataset, DatasetDict, concatenate_datasets

from transformers import BertTokenizerFast, RobertaTokenizerFast

import spacy
from spacy.symbols import ORTH
nlp = spacy.load("en_core_web_sm", disable=["tagger", "ner"])

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def tokenize_and_tag(nlp, sample, include_ppi_label=False):
	
	sent = sample['sents']

	# debug
	if re.search('[a-zA-Z]', sent) == False: # ignore sentences having no alphabets.
		print('No alphabet in the sentence:', sent)
		input('enter...')
		return
	
	# debug
	if '[E1]' not in sent or '[/E1]' not in sent or '[E2]' not in sent or '[/E2]' not in sent:
		print('No entity markers:', sent)
		input('enter...')
		return
	
	'''
	if include_ppi_label == True:
		# to make spaCy add_special_case (not splitting special tokens) work properly, special tokens need a leading and trailing whitespace.
		sent = sent.replace('[E1]', ' [E1] ').replace('[/E1]', ' [/E1] ').replace('[E2]', ' [E2] ').replace('[/E2]', ' [/E2] ')
		
		e1_s = sent.index("[E1]") + 5
		e1_e = sent.index("[/E1]") - 1
		e2_s = sent.index("[E2]") + 5
		e2_e = sent.index("[/E2]") - 1
		
		e1 = sent[e1_s:e1_e] # debug
		e2 = sent[e2_s:e2_e] # debug
		
		nlp.tokenizer.add_special_case("[E1]", [{"ORTH": "[E1]"}])
		nlp.tokenizer.add_special_case("[/E1]", [{"ORTH": "[/E1]"}])
		nlp.tokenizer.add_special_case("[E2]", [{"ORTH": "[E2]"}])
		nlp.tokenizer.add_special_case("[/E2]", [{"ORTH": "[/E2]"}])
	else:
	'''
	e1_s = sent.index("[E1]") + 5
	e1_e = sent.index("[/E1]") - 1
	e2_s = sent.index("[E2]") + 5
	e2_e = sent.index("[/E2]") - 1
	
	e1 = sent[e1_s:e1_e] # debug
	e2 = sent[e2_s:e2_e] # debug
	
	sent = sent.replace('[E1]', '').replace('[/E1]', '').replace('[E2]', '').replace('[/E2]', '')
	
	# if entity 1 appears before entity 2 in the sentence,
	if e1_s < e2_s:
		e1_s -= 4	# 4 is the length of '[E1]'
		e1_e -= 4	# 4 is the length of '[E1]'
		e2_s -= 13 	# 13 is the length of '[E1][/E1][E2]'
		e2_e -= 13 	# 13 is the length of '[E1][/E1][E2]'
	# if entity 2 appears before entity 1 in the sentence,
	else: 
		e1_s -= 13
		e1_e -= 13
		e2_s -= 4
		e2_e -= 4
	
	# debug
	n1 = sent[e1_s:e1_e] # debug
	n2 = sent[e2_s:e2_e] # debug
	if e1 != n1 or e2 != n2: # they must be the same.
		print('e1:', e1, '/ e2:', e2)
		print('n1:', n1, '/ n2:', n2)
		input('enter...')

	entity_indice = [[e1_s, e1_e], [e2_s, e2_e]]
	
	words, ner = [], []

	for tok in nlp(sent):
		token = tok.text
		t_s = tok.idx # token start index
		t_e = tok.idx + len(tok) # token end index

		if len(token.strip()) == 0: # skip empty tokens.
			continue

		# debug
		assert token == sent[t_s:t_e], 'two tokens must be the same!!'

		tag = 'O'
		for entity_idx in entity_indice:
			l_s = entity_idx[0] # label start index
			l_e = entity_idx[1] # label end index
			
			if t_s == l_s:
				tag = 'B-PROT'
				break
			elif t_s > l_s and t_s < l_e:
				tag = 'I-PROT'
				break
		
		'''
		from 'data_format_converter.py' in NER
		this is a label error. there are two ways to handle this.
		it might be better to fix the errors than ignore them.
		1) fix token using label index. e.g., Anxa6-transfected -> Anxa6, TuXoo -> Tu
		   - ignore the remaining tail texts since many of them are garbage due to text parsing error. 
		   - keep the remaining texts 04-26-2021
		2) ignore this sentence.
		'''
		if tag != 'O' and t_e > l_e:
			entity_token = sent[t_s:l_e] # revised token
			words.append(entity_token.strip())
			ner.append(tag)
			
			trailing_token = sent[l_e:t_e] # the remaining token
			words.append(trailing_token.strip())
			ner.append('O')	
		else:
			words.append(token.strip())
			ner.append(tag)
	
	#key = '-'.join([str(x) for x in list(np.concatenate(entity_indice).flat)])
	#key = e1 + '-' + e2
	
	entity_span = [] # entity indices in ner list. this is useful to find entity spans after words are tokenized by huggingface tokenizer.
	
	prev_tag = None
	for idx, tag in enumerate(ner):
		if tag == 'B-PROT':
			entity_span.append(idx)
			if prev_tag in ['B-PROT', 'I-PROT']:
				entity_span.append(idx)
			elif idx == len(ner)-1:
				entity_span.append(idx)
		elif tag == 'I-PROT' and idx == len(ner)-1:
				entity_span.append(idx)
		elif tag == 'O' and prev_tag in ['B-PROT', 'I-PROT']:
			entity_span.append(idx-1)
		prev_tag = tag
	
	# debug
	if len(entity_span) != 4:
		print(sample['sents'])
		print(sent)
		print(words)
		print(ner)
		print(entity_indice)
		print('n1:', n1, '/ n2:', n2)
		print(entity_span)
		input('enter..')

	key = '-'.join([str(x) for x in entity_span])
	
	# TODO: find a better way than manual coding.
	'''
	enzyme 0
	structural 1
	negative 2
	'''
	convert_id_to_label = ['enzyme', 'structural', 'negative']
	ppi_relation = {key: convert_id_to_label[sample['relations_id']]} # arrow_dataset.Dataset can't handle integer value when converting.
	#ppi_relation = {key: row['relations_id']}

	# debug	
	if ''.join([x for x in sent.split()]) != ''.join(words):
		print('original sent:', sample['sents'])
		print('modified sent:', sent)
		print('nlp(sent):', nlp(sent))
		print('words:', words)
		print('ner:', ner)
		print('ppi_relation:', ppi_relation)
		input('enter..')

	return words, ner, ppi_relation
	

'''
# deprecated
def convert_ppi_into_ner_format(df_ppi, include_ppi_label=False):
	
	if include_ppi_label == True:
		df_ner = pd.DataFrame(columns = ['words', 'ner', 'ppi_relation'])
	else:
		df_ner = pd.DataFrame(columns = ['words', 'ner'])
		
	for i, row in df_ppi.iterrows():
		words, ner, _ = tokenize_and_tag(nlp, row, include_ppi_label)

		if include_ppi_label == True:
			df_ner = df_ner.append({'words': words, 'ner': ner, 'ppi_relation': row['relations_id']}, ignore_index=True)	
		else:
			df_ner = df_ner.append({'words': words, 'ner': ner}, ignore_index=True)	

	return df_ner
'''

def convert_ppi_into_joint_ner_ppi_format(df_ppi):
	"""
	this function is different from convert_ppi_into_ner_format in that this integrates all NER tags and PPI relations for the same sentence.
	this removes entity markers from sentences.
	
	TODO: delete convert_ppi_into_ner_format function if not needed later.
	"""	
	data_dict = {}
	
	"""
	Collect all entity indice per sentence, which will be used to reconstruct the same sentence for different set of relations.
	Entity indice are calculated from the original sentence that doesn't have entity markers. 
	Without entity markers stripped, entity positions can't be correctly identified across different formats of the same sentence (the same sentence but different because of entity markers).
	Indice are used to add a leading and trailing space of entities, which help spaCy tokenizer properly tokenize a sentence. E.g., the a/CUB and b/coagulation factor -> the a/ CUB  and b/ coagulation  factor
	
	e.g., < different formats of the same sentence >
		Overexpression of [E1]FADD[/E1] in MCF7 and BJAB cells induces apoptosis, which, like Fas-induced apoptosis, is blocked by CrmA, a specific inhibitor of the [E2]interleukin-1 beta-converting enzyme[/E2].
		Overexpression of FADD in MCF7 and BJAB cells induces apoptosis, which, like [E1]Fas[/E1]-induced apoptosis, is blocked by [E2]CrmA[/E2], a specific inhibitor of the interleukin-1 beta-converting enzyme.
		Overexpression of FADD in MCF7 and BJAB cells induces apoptosis, which, like Fas-induced apoptosis, is blocked by [E1]CrmA[/E1], a specific inhibitor of the [E2]interleukin-1 beta[/E2]-converting enzyme.
	"""
	for i, row in df_ppi.iterrows(): # columns = ['sent_ids', 'sents', 'relations', 'relations_id']
		sent_id = row['sent_ids']
		sent = row['sents']
		
		e1_s = sent.index("[E1]") + 4
		e1_e = sent.index("[/E1]")
		e2_s = sent.index("[E2]") + 4
		e2_e = sent.index("[/E2]")
		
		# if entity 1 appears before entity 2 in the sentence,
		if e1_s < e2_s:
			e1_s -= 4	# 4 is the length of '[E1]'
			e1_e -= 4	# 4 is the length of '[E1]'
			e2_s -= 13 	# 13 is the length of '[E1][/E1][E2]'
			e2_e -= 13 	# 13 is the length of '[E1][/E1][E2]'
		# if entity 2 appears before entity 1 in the sentence,
		else: 
			e1_s -= 13
			e1_e -= 13
			e2_s -= 4
			e2_e -= 4

		if sent_id in data_dict:
			data_dict[sent_id]['data'].append(row)
			data_dict[sent_id]['all_entity_indice'].update([e1_s, e1_e, e2_s, e2_e])
		else:
			data_dict[sent_id] = {'data': [row], 'all_entity_indice': {e1_s, e1_e, e2_s, e2_e}}
	
	df_joint_ner_ppi = pd.DataFrame(columns = ['words', 'ner', 'ppi_relation'])
	
	for sent_id, v in data_dict.items():
		words_of_sent = [] # this must be one.
		ner_list = []
		ppi_list = []
		
		for sample in v['data']: # a group of relations for a sentence.
			e1_s = sample['sents'].index("[E1]") + 4
			e1_e = sample['sents'].index("[/E1]")
			e2_s = sample['sents'].index("[E2]") + 4
			e2_e = sample['sents'].index("[/E2]")
		
			sample['sents'] = sample['sents'].replace('[E1]', '').replace('[/E1]', '').replace('[E2]', '').replace('[/E2]', '')
		
			# if entity 1 appears before entity 2 in the sentence,
			if e1_s < e2_s:
				e1_s -= 4	# 4 is the length of '[E1]'
				e1_e -= 4	# 4 is the length of '[E1]'
				e2_s -= 13 	# 13 is the length of '[E1][/E1][E2]'
				e2_e -= 13 	# 13 is the length of '[E1][/E1][E2]'
			# if entity 2 appears before entity 1 in the sentence,
			else: 
				e1_s -= 13
				e1_e -= 13
				e2_s -= 4
				e2_e -= 4
			
			sample['entity_indice'] = [e1_s, e1_e, e2_s, e2_e]

			for a_e_idx in sorted(v['all_entity_indice'], reverse=True): # add a space from backwards to preserve earlier indice.
				sample['sents'] = sample['sents'][:a_e_idx] + ' ' + sample['sents'][a_e_idx:]

				for j, e_idx in enumerate(sample['entity_indice']):
					if e_idx >= a_e_idx:
						sample['entity_indice'][j] += 1
			
			new_e1_s = sample['entity_indice'][0]
			new_e1_e = sample['entity_indice'][1]
			new_e2_s = sample['entity_indice'][2]
			new_e2_e = sample['entity_indice'][3]
			
			# if entity 1 appears before entity 2 in the sentence,
			if new_e1_s < new_e2_s:
				sample['sents'] = sample['sents'][:new_e2_e] + '[/E2]' + sample['sents'][new_e2_e:]
				sample['sents'] = sample['sents'][:new_e2_s] + '[E2] ' + sample['sents'][new_e2_s:] # add a space after an entity start marker
				sample['sents'] = sample['sents'][:new_e1_e] + '[/E1]' + sample['sents'][new_e1_e:]
				sample['sents'] = sample['sents'][:new_e1_s] + '[E1] ' + sample['sents'][new_e1_s:] # add a space after an entity start marker
			# if entity 2 appears before entity 1 in the sentence,
			else:
				sample['sents'] = sample['sents'][:new_e1_e] + '[/E1]' + sample['sents'][new_e1_e:]
				sample['sents'] = sample['sents'][:new_e1_s] + '[E1] ' + sample['sents'][new_e1_s:] # add a space after an entity start marker
				sample['sents'] = sample['sents'][:new_e2_e] + '[/E2]' + sample['sents'][new_e2_e:]
				sample['sents'] = sample['sents'][:new_e2_s] + '[E2] ' + sample['sents'][new_e2_s:] # add a space after an entity start marker

			# Caution!! don't replace multiple whitespaces to a single space because that breaks entity indice.
			
			words, ner, ppi = tokenize_and_tag(nlp, sample)
			
			# debug - the parsing results must be the same because they are the same sentence. check if they are the same.
			if len(words_of_sent) > 0 and words != words_of_sent:
				print('Error - parsing results for the same sentence are different!!')
				print('sent_id:', sent_id)
				print('words_of_sent:', words_tmp)
				print('words:', words)
				input('enter...')
				
			if not words_of_sent:
				words_of_sent = words
			ner_list.append(ner)
			ppi_list.append(ppi)
				
		combined_ner = None
		for ner in ner_list:
			if combined_ner is not None:
				old_ner = combined_ner
				new_ner = ner
				# update NER
				combined_ner = [x if x == new_ner[i] or x != 'O' else new_ner[i] for i, x in enumerate(old_ner)]
				
				# debug
				if len(old_ner) != len(new_ner) != len(combined_ner):
					print('length error!!')
					print('sent_id:', sent_id)
					print('sent:', sample['sents'])
					print('words_of_sent:', words_of_sent)
					print('len(old_ner):', len(old_ner))
					print('len(new_ner):', len(new_ner))
					print('len(combined_ner):', len(combined_ner))
					print('old_ner:', old_ner)
					print('new_ner:', new_ner)
					print('combined_ner:', combined_ner)
					input('enter..')
			else:
				combined_ner = ner
	
		combined_ppi = None
		for ppi in ppi_list:
			if combined_ppi is not None:
				old_ppi = combined_ppi
				new_ppi = ppi
				# update PPI
				if list(new_ppi.keys())[0] not in old_ppi: # avoid duplicates.
					old_ppi.update(new_ppi)
				'''
				else:
					# debug
					# One error exists in the file: BioInfer has the duplicates for the following.
					#	<entity charOffset="176-179" id="BioInfer.d109.s1.e4" origId="e.246.9" text="NRP1" type="Individual_protein" />
					#	<entity charOffset="176-179" id="BioInfer.d109.s1.e6" origId="e.246.11" text="NRP1" type="Individual_protein" />
					print('duplicate ppi relation error!!')
					print('sent_id:', sent_id)
					print('sent:', sample['sents'])
					print('words_of_sent:', words_of_sent)
					print('old_ppi:', old_ppi)
					print('new_ppi:', new_ppi)
					input('enter...')
				'''
			else:
				combined_ppi = ppi

		combined_ppi = [[k, v] for k, v in combined_ppi.items()] # arrow_dataset.Dataset can't handle dictionary type for dictionary value when converting data using from_pandas or from_dict.
		df_joint_ner_ppi = df_joint_ner_ppi.append({'sent_id': sent_id, 'words': words_of_sent, 'ner': combined_ner, 'ppi_relation': combined_ppi}, ignore_index=True)
	
	
	# debug
	'''
	# check the total number of PPIs.
	print('Before - # PPI relations:', df_ppi.shape[0])
	after_cnt = 0
	for i, row in df_joint_ner_ppi.iterrows():
		after_cnt += len(row['ppi_relation'])
	print('After  - # PPI relations:', after_cnt)
	
	old_sent_ppi_dict = {}
	for i, row in df_ppi.iterrows():
		sent_id = row['sent_ids']
		relations_id = row['relations_id']
		
		if sent_id not in old_sent_ppi_dict:
			old_sent_ppi_dict[sent_id] = [0, 0, 0]

		if relations_id == 0:
			old_sent_ppi_dict[sent_id][0] += 1
		elif relations_id == 1:
			old_sent_ppi_dict[sent_id][1] += 1
		elif relations_id == 2:
			old_sent_ppi_dict[sent_id][2] += 1
	
	# check the number of PPIs per each class per each sentence.
	# there must be only one different due to the "BioInfer.d109.s1.e4" and "BioInfer.d109.s1.e6"
	new_sent_ppi_dict = {}
	for i, row in df_joint_ner_ppi.iterrows():
		sent_id = row['sent_id']
		relations = row['ppi_relation']
		
		if sent_id not in new_sent_ppi_dict:
			new_sent_ppi_dict[sent_id] = [0, 0, 0]
			
		for rel in relations:
			if rel[1] == 'enzyme':
				new_sent_ppi_dict[sent_id][0] += 1
			elif rel[1] == 'structural':
				new_sent_ppi_dict[sent_id][1] += 1
			elif rel[1] == 'negative':
				new_sent_ppi_dict[sent_id][2] += 1
		
	for k, v in new_sent_ppi_dict.items():
		if v != old_sent_ppi_dict[k]:
			print('sent_id:', k)
			print('old ppi:', old_sent_ppi_dict[k])
			print('new ppi:', v)
			input('enter...')
	'''
	
	return df_joint_ner_ppi


def read_dataset(dataset_num=0, task_name=None, data_args=None):
	
	if task_name == 'ner':
		data_dir = data_args.ner_data
		
		file_type = None
		for file in os.listdir(data_dir):
			if file.endswith('.pkl'):
				file_type = 'pkl'
				break
			elif file.endswith('.json'):
				file_type = 'json'
				break
		'''
		# TODO: change to a better condition.
		if data_dir == data_args.ppi_data: # if NER data is from PPI data,
			
			train_path = os.path.join(data_dir, 'df_train_' + str(dataset_num) + '.pkl')
			dev_path = os.path.join(data_dir, 'df_dev_' + str(dataset_num) + '.pkl')
			test_path = os.path.join(data_dir, 'df_test_' + str(dataset_num) + '.pkl')
			
			assert os.path.isfile(train_path) and os.path.isfile(test_path)		
			
			df_train = pickle.load(open(train_path, "rb"))
			df_dev = pickle.load(open(dev_path, "rb")) if os.path.isfile(dev_path) else None # optional
			df_test = pickle.load(open(test_path, "rb"))

			data_dict = DatasetDict()
			
			# TODO: remove joint-ner-ppi!!
			if task_name == 'joint-ner-ppi':		
				data_dict["train"] = Dataset.from_pandas(convert_ppi_into_ner_format(df_train, include_ppi_label=True))
				data_dict["test"] = Dataset.from_pandas(convert_ppi_into_ner_format(df_test, include_ppi_label=True))
				if df_dev is not None:
					data_dict["validation"] = Dataset.from_pandas(convert_ppi_into_ner_format(df_dev, include_ppi_label=True))
			else:
				data_dict["train"] = Dataset.from_pandas(convert_ppi_into_ner_format(df_train, include_ppi_label=False))
				data_dict["test"] = Dataset.from_pandas(convert_ppi_into_ner_format(df_test, include_ppi_label=False))
				if df_dev is not None:
					data_dict["validation"] = Dataset.from_pandas(convert_ppi_into_ner_format(df_dev, include_ppi_label=False))
			
			return data_dict
		'''
		if file_type == 'pkl':
			train_path = os.path.join(data_dir, 'df_train_' + str(dataset_num) + '.pkl')
			dev_path = os.path.join(data_dir, 'df_dev_' + str(dataset_num) + '.pkl')
			test_path = os.path.join(data_dir, 'df_test_' + str(dataset_num) + '.pkl')

			df_train = pickle.load(open(train_path, "rb"))
			df_dev = pickle.load(open(dev_path, "rb")) if os.path.isfile(dev_path) else None # optional
			df_test = pickle.load(open(test_path, "rb"))	
			
			data_dict = DatasetDict()
			data_dict["train"] = Dataset.from_pandas(df_train)
			data_dict["test"] = Dataset.from_pandas(df_test)
			if df_dev is not None:
				data_dict["validation"] = Dataset.from_pandas(df_dev)
			
			
		
			'''
			# [START] temporary code for adding BioCreative NER datasets. 05-09-2021
			# to concatenate both datasets must have the same features.
			data_dict["train"] = data_dict["train"].remove_columns(["sent_id", "ppi_relation"])
			data_dict["test"] = data_dict["test"].remove_columns(["sent_id", "ppi_relation"])
			
			biocreative_ner_dir = '/sdcc/u/gpark/BER-NLP/Multi-Task-Learning/datasets/ner/BioCreative_I_Gene_Mention_Identification_10_fold_cv_no_test_data'
			
			data_files = {}
			data_files["train"] = os.path.join(biocreative_ner_dir, 'train_' + str(dataset_num) + '.json')
			data_files["test"] = os.path.join(biocreative_ner_dir, 'test_' + str(dataset_num) + '.json')
			
			biocreative_ner_data = load_dataset('json', data_files=data_files)
			
			data_dict["train"] = concatenate_datasets([data_dict["train"], biocreative_ner_data["train"]])
			data_dict["test"] = concatenate_datasets([data_dict["test"], biocreative_ner_data["test"]])
			# [END] temporary code for adding BioCreative NER datasets. 05-09-2021
			'''
			
			

			return data_dict
			
		elif file_type == 'json':
			assert data_dir != None
			
			data_files = {}
			data_files["train"] = os.path.join(data_dir, 'train_' + str(dataset_num) + '.json')
			data_files["test"] = os.path.join(data_dir, 'test_' + str(dataset_num) + '.json')
			if os.path.isfile(os.path.join(data_dir, 'dev_' + str(dataset_num) + '.json')):
				data_files["validation"] = os.path.join(data_dir, 'dev_' + str(dataset_num) + '.json')
				
			extension = data_files["train"].split(".")[-1]
			
			return load_dataset(extension, data_files=data_files)
		
		else:
			logger.error("Unknown data type!!")
	
	elif task_name == 'ppi':
		data_dir = data_args.ppi_data
			
		data_files = {}
		data_files["train"] = os.path.join(data_dir, 'train_' + str(dataset_num) + '.json')
		data_files["test"] = os.path.join(data_dir, 'test_' + str(dataset_num) + '.json')
		if os.path.isfile(os.path.join(data_dir, 'dev_' + str(dataset_num) + '.json')):
			data_files["validation"] = os.path.join(data_dir, 'dev_' + str(dataset_num) + '.json')
			
		extension = data_files["train"].split(".")[-1]
		
		return load_dataset(extension, data_files=data_files)
		
		""" old code
		elif task_name == 'ppi':
			data_dir = data_args.ppi_data
				
			#relations_path = os.path.join(data_dir, 'relations_' + str(dataset_num) + '.pkl')
			train_path = os.path.join(data_dir, 'df_train_' + str(dataset_num) + '.pkl')
			dev_path = os.path.join(data_dir, 'df_dev_' + str(dataset_num) + '.pkl')
			test_path = os.path.join(data_dir, 'df_test_' + str(dataset_num) + '.pkl')
			
			#assert os.path.isfile(relations_path) and os.path.isfile(train_path) and os.path.isfile(test_path)
			assert os.path.isfile(train_path) and os.path.isfile(test_path)
			
			#rm = pickle.load(open(relations_path, "rb"))
			df_train = pickle.load(open(train_path, "rb"))
			df_dev = pickle.load(open(dev_path, "rb")) if os.path.isfile(dev_path) else None # optional
			df_test = pickle.load(open(test_path, "rb"))


			
			# [START] Test code
			'''
			- double the size of data to compare with MTL. 05-17-2021
			- This is to check if simply doubled PPI data performed better than MTL that has the same data size (NER + PPI).
			- the result shows that MTL performed better than STL with the doubled size PPI data.
			'''
			'''
			df_train = pd.concat([df_train, df_train])
			df_test = pd.concat([df_test, df_test])
			
			# debugging
			#pd.set_option('display.max_rows', None)
			#pd.set_option('display.max_columns', None)
			#pd.set_option('display.width', None)
			#pd.set_option('display.max_colwidth', -1)
			#for idx, row in df_train.iterrows():
			#	if row['sents'].startswith('Mutagenesis studies of the human'):
			#		print(row['sents'])
			#		input('enter..')
			'''
			# [END] Test code
			
			
			data_dict = DatasetDict()

			data_dict["train"] = Dataset.from_pandas(df_train)
			data_dict["test"] = Dataset.from_pandas(df_test)
			if df_dev is not None:
				data_dict["validation"] = Dataset.from_pandas(df_dev)
			
			if data_args.relation_representation == 'STANDARD_cls_token':
				def remove_entity_marker(example):
					example['sents'] = example['sents'].replace("[E1]", '').replace("[/E1]", '').replace("[E2]", '').replace("[/E2]", '')
					return example
			
				data_dict["train"] = data_dict["train"].map(remove_entity_marker)
				data_dict["test"] = data_dict["test"].map(remove_entity_marker)
				if df_dev is not None:
					data_dict["validation"] = data_dict["validation"].map(remove_entity_marker)

			# debug
			'''
			for i in data_dict["train"]:
				if not ("[E1]" or "[E2]") in i['sents']:
					print(i)
					input('enter...')		
			'''
			
			return data_dict
		"""
	elif task_name == 'joint-ner-ppi':
		data_dir = data_args.joint_ner_ppi_data

		train_path = os.path.join(data_dir, 'df_train_' + str(dataset_num) + '.pkl')
		dev_path = os.path.join(data_dir, 'df_dev_' + str(dataset_num) + '.pkl')
		test_path = os.path.join(data_dir, 'df_test_' + str(dataset_num) + '.pkl')
		
		
		
		# temporarily loading ppi test set in dev path for PPI prediction. 05-02-2021
		dev_path = os.path.join(data_args.ppi_data, 'df_test_' + str(dataset_num) + '.pkl')
		
		

		if os.path.isfile(train_path) and os.path.isfile(test_path):
			df_train = pickle.load(open(train_path, "rb"))
			df_dev = pickle.load(open(dev_path, "rb")) if os.path.isfile(dev_path) else None # optional
			df_test = pickle.load(open(test_path, "rb"))	
		else:
			ppi_data_dir = data_args.ppi_data # TODO: find a better way!!
		
			train_path = os.path.join(ppi_data_dir, 'df_train_' + str(dataset_num) + '.pkl')
			dev_path = os.path.join(ppi_data_dir, 'df_dev_' + str(dataset_num) + '.pkl')
			test_path = os.path.join(ppi_data_dir, 'df_test_' + str(dataset_num) + '.pkl')

			assert os.path.isfile(train_path) and os.path.isfile(test_path)		
		
			df_train = pickle.load(open(train_path, "rb"))
			df_dev = pickle.load(open(dev_path, "rb")) if os.path.isfile(dev_path) else None # optional
			df_test = pickle.load(open(test_path, "rb"))
			
			df_train = convert_ppi_into_joint_ner_ppi_format(df_train)
			df_test = convert_ppi_into_joint_ner_ppi_format(df_test)
			if df_dev is not None:
				df_dev = convert_ppi_into_joint_ner_ppi_format(df_dev)
			
			if not os.path.exists(data_dir):
				os.makedirs(data_dir)

			pickle.dump(df_train, open(os.path.join(data_dir, 'df_train_' + str(dataset_num) + '.pkl'), "wb"))
			pickle.dump(df_test, open(os.path.join(data_dir, 'df_test_' + str(dataset_num) + '.pkl'), "wb"))
			if df_dev is not None:
				pickle.dump(df_dev, open(os.path.join(data_dir, 'df_dev_' + str(dataset_num) + '.pkl'), "wb"))

		data_dict = DatasetDict()

		data_dict["train"] = Dataset.from_pandas(df_train)
		data_dict["test"] = Dataset.from_pandas(df_test)
		if df_dev is not None:
			data_dict["validation"] = Dataset.from_pandas(df_dev)

		return data_dict


# In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
# unique labels.
def get_label_list(labels):
	unique_labels = set()
	for label in labels:
		unique_labels = unique_labels | set(label)
	label_list = list(unique_labels)
	label_list.sort()
	return label_list


def get_num_of_labels(task_name=None, dataset_dict=None, training_args=None, data_args=None):
	if task_name == 'ner' or task_name == 'joint-ner-ppi':
		
		## [start] from NER
		
		if training_args.do_train:
			column_names = dataset_dict[task_name]["train"].column_names
			features = dataset_dict[task_name]["train"].features
		else:
			column_names = dataset_dict[task_name]["validation"].column_names
			features = dataset_dict[task_name]["validation"].features
		text_column_name = "tokens" if "tokens" in column_names else column_names[0]
		label_column_name = (
			"ner_tags" if "ner_tags" in column_names else column_names[1]
		)

		if isinstance(features[label_column_name].feature, ClassLabel):
			label_list = features[label_column_name].feature.names
			# No need to convert the labels since they are already ints.
			label_to_id = {i: i for i in range(len(label_list))}
		else:
			label_list = get_label_list(dataset_dict[task_name]["train"][label_column_name])
			label_to_id = {l: i for i, l in enumerate(label_list)}
		num_labels = len(label_list)
		
		# debug
		'''
		print('text_column_name:', text_column_name)
		print('label_column_name:', label_column_name)
		print('label_list:', label_list)
		print('num_labels:', num_labels)
		#input('enter...')	
		'''
		
		return num_labels
		
		## [end] from NER
		
		
	elif task_name == 'ppi':
		return len(json.load(open(data_args.relations)))


## [start] from NER

# Tokenize all texts and align the labels with them.
def tokenize_and_align_labels(examples, tokenizer, text_column_name, label_column_name, label_to_id, label_all_tokens, padding):
	tokenized_inputs = tokenizer(
		examples[text_column_name],
		padding=padding,
		truncation=True,
		# We use this argument because the texts in our dataset are lists of words (with a label for each word).
		is_split_into_words=True,	
	)
	labels = []
	for i, label in enumerate(examples[label_column_name]):
		word_ids = tokenized_inputs.word_ids(batch_index=i)
		previous_word_idx = None
		label_ids = []
		for word_idx in word_ids:
			# Special tokens have a word id that is None. We set the label to -100 so they are automatically
			# ignored in the loss function.
			if word_idx is None:
				label_ids.append(-100)
			# We set the label for the first token of each word.
			elif word_idx != previous_word_idx:
				label_ids.append(label_to_id[label[word_idx]])
			# For the other tokens in a word, we set the label to either the current label or -100, depending on
			# the label_all_tokens flag.
			else:
				label_ids.append(label_to_id[label[word_idx]] if label_all_tokens else -100)
			previous_word_idx = word_idx

		labels.append(label_ids)
	tokenized_inputs['labels'] = labels

	return tokenized_inputs

## [end] from NER


def get_e1e2_start(tokenizer, x, e1_start_id, e2_start_id):
	try:
		e1_e2_start = ([i for i, e in enumerate(x) if e == e1_start_id][0],\
					   [i for i, e in enumerate(x) if e == e2_start_id][0])
	except Exception as e:
		e1_e2_start = None
		print("<get_e1e2_start()> error msg:", e)
		print("<get_e1e2_start()> input ids:", x)
		print("<get_e1e2_start()> input tokens:", [tokenizer.convert_ids_to_tokens(i) for i in x])
		print("<get_e1e2_start()> e1_start_id:", e1_start_id, " / e2_start_id:", e2_start_id)

	return e1_e2_start


def get_entity_mention(tokenizer, x, e1_start_id, e2_start_id, e1_end_id, e2_end_id):
	try:
		entity_mention = ([i+1 for i, e in enumerate(x) if e == e1_start_id][0],\
						  [i for i, e in enumerate(x) if e == e1_end_id][0],\
						  [i+1 for i, e in enumerate(x) if e == e2_start_id][0],\
						  [i for i, e in enumerate(x) if e == e2_end_id][0])
	except Exception as e:
		entity_mention = None
		print("<get_entity_mention()> error msg:", e)
		print("<get_entity_mention()> input ids:", x)
		print("<get_entity_mention()> input tokens:", [tokenizer.convert_ids_to_tokens(i) for i in x])
		print("<get_entity_mention()> e1_start_id:", e1_start_id, "/ e1_end_id:", e1_end_id, "/ e2_start_id:", e2_start_id, " / e2_end_id:", e2_end_id)

	return entity_mention

		   
def tokenize_and_find_em(examples, tokenizer, relation_representation, e1_start_id, e2_start_id, e1_end_id, e2_end_id, padding):

	tokenized_inputs = tokenizer(
		examples['entity_marked_sent'],
		padding=padding,
		truncation=True,
	)
	
	# debug
	'''
	print(tokenizer)
	print(isinstance(tokenizer, BertTokenizerFast))
	for k, v in tokenized_inputs.items():
		print(k, v)
		
	input('enter..')
	'''
	
	''' 
	print(tokenizer.additional_special_tokens)	
	for i in tokenized_inputs['input_ids']:
		print(tokenizer.convert_ids_to_tokens(i))
	print('e1_id:', e1_id, ' / e2_id:', e2_id)
	'''

	if relation_representation in ['EM_entity_start', 'EM_entity_start_plus_context']:
		tokenized_inputs['e1_e2_start'] = list(map(lambda x: get_e1e2_start(tokenizer, x, e1_start_id=e1_start_id, e2_start_id=e2_start_id), tokenized_inputs['input_ids']))
		#tokenized_inputs['e1_e2_start'] = self.df.progress_apply(lambda x: get_e1e2_start(x['input'], e1_id=self.e1_id, e2_id=self.e2_id), axis=1)
	
	if relation_representation in ['STANDARD_mention_pooling', 'STANDARD_mention_pooling_plus_context', \
								   'EM_mention_pooling', 'EM_entity_start_plus_context']:
		tokenized_inputs['entity_mention'] = list(map(lambda x: get_entity_mention(tokenizer, x, e1_start_id=e1_start_id, e2_start_id=e2_start_id, e1_end_id=e1_end_id, e2_end_id=e2_end_id), tokenized_inputs['input_ids']))

		if relation_representation in ['STANDARD_mention_pooling', 'STANDARD_mention_pooling_plus_context']: # remove entity markers and adjust entity indices.
			entity_mention = tokenized_inputs['entity_mention']
			entity_mention = [list(x) for x in entity_mention]
			
			adjusted_entity_mention = []
			for idx, elem in enumerate(entity_mention): # [e1_start_idx, e1_end_idx, e2_start_idx, e2_end_idx]
				e1_start_idx = elem[0]
				e1_end_idx = elem[1]
				e2_start_idx = elem[2]
				e2_end_idx = elem[3]
				
				# remove entity markers from the high index number to preserve the other indices. [E1] is start idx - 1.
				for e_idx in sorted([e1_start_idx - 1, e1_end_idx, e2_start_idx - 1, e2_end_idx], reverse=True):
					tokenized_inputs['input_ids'][idx].pop(e_idx)
					if isinstance(tokenizer, RobertaTokenizerFast) == False:
						tokenized_inputs['token_type_ids'][idx].pop(e_idx)
					tokenized_inputs['attention_mask'][idx].pop(e_idx)
					
				# if entity 1 appears before entity 2 in the sentence,
				if e1_start_idx < e2_start_idx:
					e1_start_idx = e1_start_idx - 1
					e1_end_idx = e1_end_idx - 1
					e2_start_idx = e2_start_idx - 3
					e2_end_idx = e2_end_idx - 3
				# if entity 2 appears before entity 1 in the sentence,
				else: 
					e1_start_idx = e1_start_idx - 3
					e1_end_idx = e1_end_idx - 3
					e2_start_idx = e2_start_idx - 1
					e2_end_idx = e2_end_idx - 1
										
				adjusted_entity_mention.append((e1_start_idx, e1_end_idx, e2_start_idx, e2_end_idx))

			tokenized_inputs['entity_mention'] = adjusted_entity_mention

	tokenized_inputs['labels'] = np.reshape(examples['relation_id'], (-1, 1)).tolist() # convert format. e.g., [0, 0] -> [[0], [0]]
	
	# debug
	''' 
	for i in tokenized_inputs:
		print(i)
		
	print(tokenized_inputs['input_ids'])
	for idx, elem in enumerate(tokenized_inputs['input_ids']):
		print(tokenizer.convert_ids_to_tokens(elem))
		print(tokenized_inputs['token_type_ids'])
		print(tokenized_inputs['attention_mask'])
		
	print(tokenized_inputs['labels'])
	print(tokenized_inputs['e1_e2_start'])
	print('e1_id:', e1_id)
	print('e2_id:', e2_id)
	input('enter..')
	'''
	
	return tokenized_inputs


def tokenize_and_align_labels_and_find_em(examples, tokenizer, text_column_name, label_column_name, label_to_id, label_all_tokens, \
										  relation_representation, e1_start_id, e2_start_id, e1_end_id, e2_end_id, padding, phase):
	tokenized_inputs = tokenizer(
		examples[text_column_name],
		padding=padding,
		truncation=True,
		# We use this argument because the texts in our dataset are lists of words (with a label for each word).
		is_split_into_words=True,	
	)
	
	labels = [] # NER labels
	ppi_relations = []
	max_ppi_span_len = 0
	
	for i, label in enumerate(examples[label_column_name]):
		word_ids = tokenized_inputs.word_ids(batch_index=i)
		previous_word_idx = None
		label_ids = []
		
		ppi_relation = examples['ppi_relation'][i]	
		ppi_entity_span = []
		
		ppi_unique_set = set() # debug - to check if there are duplicate set that just have different direction. e.g., ((36, 39), (27, 6))), ((27, 6), (36, 39))
		
		for x in ppi_relation:
			indice = [int(y) for y in x[0].split('-')]
			e1_s, e1_e, e2_s, e2_e = indice[0], indice[1], indice[2], indice[3]
			
			e1_span_s = word_ids.index(e1_s)
			e1_span_e = len(word_ids) - word_ids[::-1].index(e1_e)
			e2_span_s = word_ids.index(e2_s)
			e2_span_e = len(word_ids) - word_ids[::-1].index(e2_e)
			
			# debug
			e1_old = examples['words'][i][e1_s] if e1_s == e1_e else examples['words'][i][e1_s:e1_e+1]
			e2_old = examples['words'][i][e2_s] if e2_s == e2_e else examples['words'][i][e2_s:e2_e+1]
			e1_old = ''.join(e1_old) if isinstance(e1_old, list) else e1_old
			e2_old = ''.join(e2_old) if isinstance(e2_old, list) else e2_old
			e1_new = tokenizer.convert_ids_to_tokens(tokenized_inputs['input_ids'][i])[e1_span_s:e1_span_e]
			e2_new = tokenizer.convert_ids_to_tokens(tokenized_inputs['input_ids'][i])[e2_span_s:e2_span_e]
			e1_new = ''.join([x.replace('##', '') for x in e1_new])
			e2_new = ''.join([x.replace('##', '') for x in e2_new])
			if e1_old != e1_new or e2_old != e2_new:
				print(examples['words'][i])
				print('e1_s:', e1_s, '/ e1_e:', e1_e, '/ e2_s:', e2_s, '/ e2_e:', e2_e)
				print('e1_old:', e1_old, '/ e2_old:', e2_old)
				print('e1_new:', e1_new, '/ e2_new:', e2_new)
				input('enter..')
						
			if tuple(sorted([e1_s, e1_e, e2_s, e2_e])) in ppi_unique_set:
				print(examples['words'][i])
				print('duplicate ppi set:', e1_s, e1_e, e2_s, e2_e)
				input('enter..')
			else:
				ppi_unique_set.add(tuple(sorted([e1_s, e1_e, e2_s, e2_e])))
			
			if e1_s > e2_s:
				print(examples['words'][i])
				print('e1_s is greater than e2_s:', e1_s, e1_e, e2_s, e2_e)
				input('enter..')

			
			# TODO: find a better way than manual coding.
			'''
			enzyme 0
			structural 1
			negative 2
			'''
			convert_label_to_id = {'enzyme': 0, 'structural': 1, 'negative': 2}
			ppi_entity_span.extend([e1_span_s, e1_span_e, e2_span_s, e2_span_e, convert_label_to_id[x[1]]])
		
		for word_idx in word_ids:
			# Special tokens have a word id that is None. We set the label to -100 so they are automatically
			# ignored in the loss function.
			if word_idx is None:
				label_ids.append(-100)
			# We set the label for the first token of each word.
			elif word_idx != previous_word_idx:
				label_ids.append(label_to_id[label[word_idx]])
			# For the other tokens in a word, we set the label to either the current label or -100, depending on
			# the label_all_tokens flag.
			else:
				label_ids.append(label_to_id[label[word_idx]] if label_all_tokens else -100)
			previous_word_idx = word_idx
			
		labels.append(label_ids)

		if len(ppi_entity_span) > max_ppi_span_len:
			max_ppi_span_len = len(ppi_entity_span)

		'''
		if len(ppi_entity_span) > 500:

			print(ppi_entity_span)
			print(examples['ppi_relation'][i])
			print(examples['words'][i])
			
			input('enter..')
		'''	

		ppi_relations.append(ppi_entity_span)
		
	tokenized_inputs['labels'] = labels

	#for i in ppi_relations:
	#	i += [-100] * (1100 - len(i)) # padding

	tokenized_inputs['ppi_relations'] = ppi_relations
	#print('max_ppi_span_len:', max_ppi_span_len)


	
	
	# temporarily closed to test joint_ner_ppi data. 04-29-2021
	''' 
	if relation_representation == 'EM_entity_start':
		tokenized_inputs['e1_e2_start'] = list(map(lambda x: get_e1e2_start(tokenizer, x, e1_start_id=e1_start_id, e2_start_id=e2_start_id), tokenized_inputs['input_ids']))
		#tokenized_inputs['e1_e2_start'] = self.df.progress_apply(lambda x: get_e1e2_start(x['input'], e1_id=self.e1_id, e2_id=self.e2_id), axis=1)
	elif relation_representation in ['STANDARD_mention_pooling', 'EM_mention_pooling']:
		tokenized_inputs['entity_mention'] = list(map(lambda x: get_entity_mention(tokenizer, x, e1_start_id=e1_start_id, e2_start_id=e2_start_id, e1_end_id=e1_end_id, e2_end_id=e2_end_id), tokenized_inputs['input_ids']))
		
		# debug
		#import copy
		#test = copy.deepcopy(tokenized_inputs)
	
		if relation_representation == 'STANDARD_mention_pooling': # remove entity markers and adjust entity indices.
			entity_mention = tokenized_inputs['entity_mention']
			entity_mention = [list(x) for x in entity_mention]
			
			adjusted_entity_mention = []
			for idx, elem in enumerate(entity_mention): # [e1_start_idx, e1_end_idx, e2_start_idx, e2_end_idx]
				e1_start_idx = elem[0]
				e1_end_idx = elem[1]
				e2_start_idx = elem[2]
				e2_end_idx = elem[3]
				
				# remove entity markers from the high index number to preserve the other indices. [E1] is start idx - 1.
				for e_idx in sorted([e1_start_idx - 1, e1_end_idx, e2_start_idx - 1, e2_end_idx], reverse=True):
					tokenized_inputs['input_ids'][idx].pop(e_idx)
					tokenized_inputs['token_type_ids'][idx].pop(e_idx)
					tokenized_inputs['attention_mask'][idx].pop(e_idx)
					tokenized_inputs['labels'][idx].pop(e_idx)
					
				# if entity 1 appears before entity 2 in the sentence,
				if e1_start_idx < e2_start_idx:
					e1_start_idx = e1_start_idx - 1
					e1_end_idx = e1_end_idx - 1
					e2_start_idx = e2_start_idx - 3
					e2_end_idx = e2_end_idx - 3
				# if entity 2 appears before entity 1 in the sentence,
				else: 
					e1_start_idx = e1_start_idx - 3
					e1_end_idx = e1_end_idx - 3
					e2_start_idx = e2_start_idx - 1
					e2_end_idx = e2_end_idx - 1
										
				adjusted_entity_mention.append((e1_start_idx, e1_end_idx, e2_start_idx, e2_end_idx))

			tokenized_inputs['entity_mention'] = adjusted_entity_mention

	tokenized_inputs['ppi_relations'] = np.reshape(examples['ppi_relation'], (-1, 1)).tolist() # convert format. e.g., [0, 0] -> [[0], [0]]
	
	# when trainer predicts samples, it uses 'labels'. 
	if phase in ['validation', 'test']:
		tokenized_inputs['labels'] = tokenized_inputs['ppi_relations']
	'''
	
	# debug
	'''	
	for k, v in examples.items():
		print('examples:', k, v)
		
	for idx, elem in enumerate(tokenized_inputs['input_ids']):	
		print('input:', tokenizer.convert_ids_to_tokens(elem))
		print('token_type_ids:', tokenized_inputs['token_type_ids'][idx])
		print('attention_mask:', tokenized_inputs['attention_mask'][idx])
		print('labels:', tokenized_inputs['labels'][idx])
		print('ppi_relations:', tokenized_inputs['ppi_relations'][idx])
		input('enter..')
		
	
	for k, v in examples.items():
		print('examples:', k, v)
	
	for idx, elem in enumerate(tokenized_inputs['input_ids']):	
		print('Before - input:', tokenizer.convert_ids_to_tokens(test['input_ids'][idx]))
		print('After  - input:', tokenizer.convert_ids_to_tokens(elem))
		print('Before - token_type_ids:', test['token_type_ids'][idx])
		print('After  - token_type_ids:', tokenized_inputs['token_type_ids'][idx])
		print('Before - attention_mask:', test['attention_mask'][idx])
		print('After  - attention_mask:', tokenized_inputs['attention_mask'][idx])
		print('Before - labels:', test['labels'][idx])
		print('After  - labels:', tokenized_inputs['labels'][idx])
		print('Before - entity_mention:', test['entity_mention'][idx])
		print('After  - entity_mention:', tokenized_inputs['entity_mention'][idx])
		input('enter..')

	print(examples['ppi_relation'])
	print(type(examples['ppi_relation']))
	print(np.reshape(examples['ppi_relation'], (-1, 1)).tolist())
	
	for i in tokenized_inputs:
		print(i)
		
	print(tokenized_inputs['input_ids'])
	for i in tokenized_inputs['input_ids']:
		print(tokenizer.convert_ids_to_tokens(i))
	print(tokenized_inputs['labels'])
	print(tokenized_inputs['e1_e2_start'])
	print('e1_id:', e1_id)
	print('e2_id:', e2_id)
	input('enter..')
	'''
	return tokenized_inputs

## [end] from NER


	
def featurize_data(dataset_dict, tokenizer_dict, padding, data_args, do_lower_case):

	convert_func_dict = {
		#"stsb": convert_to_stsb_features,
		#"rte": convert_to_rte_features,
		#"commonsense_qa": convert_to_commonsense_qa_features,
		"ner": tokenize_and_align_labels,
		"ppi": tokenize_and_find_em, # TODO: rename the funciton. this needs to be fixed and made cleaner later. EM: Entity Marker 
		"joint-ner-ppi": tokenize_and_align_labels_and_find_em, # TODO: rename the funciton. this needs to be fixed and made cleaner later.
	}

	columns_dict = {
		#"stsb": ['input_ids', 'attention_mask', 'labels'],
		#"rte": ['input_ids', 'attention_mask', 'labels'],
		#"commonsense_qa": ['input_ids', 'attention_mask', 'labels'],
		"ner": ['input_ids', 'attention_mask', 'labels', 'token_type_ids'],
		"ppi": ['input_ids', 'attention_mask', 'labels', 'token_type_ids'],
		"joint-ner-ppi": ['input_ids', 'attention_mask', 'labels', 'token_type_ids', 'ppi_relations'],
	}

	if data_args.relation_representation in ['EM_entity_start', 'EM_entity_start_plus_context']:
		columns_dict["ppi"].append('e1_e2_start')
		#columns_dict["joint-ner-ppi"].append('e1_e2_start')
	
	if data_args.relation_representation in ['STANDARD_mention_pooling', 'STANDARD_mention_pooling_plus_context', \
											 'EM_mention_pooling', 'EM_entity_start_plus_context']:
		columns_dict["ppi"].append('entity_mention')
		#columns_dict["joint-ner-ppi"].append('entity_mention')
	
	features_dict = {}
	for task_name, dataset in dataset_dict.items():
		features_dict[task_name] = {}
		tokenizer = tokenizer_dict[task_name]
		
		if isinstance(tokenizer_dict[task_name], RobertaTokenizerFast):
			columns_dict[task_name].remove('token_type_ids') # RoBERTa doesn't use NSP, so it doesn't need 'token_type_ids' which is segement IDs.
		
		# TODO: this is redundant with the code above. clean it!!
		if task_name == 'ner' or task_name == 'joint-ner-ppi':
			
			#print(type(dataset)) # <class 'datasets.dataset_dict.DatasetDict'>
			#print(type(dataset["train"])) # <class 'datasets.arrow_dataset.Dataset'>
			#input('enter..')
		
			column_names = dataset["train"].column_names
			features = dataset["train"].features

			text_column_name = "tokens" if "tokens" in column_names else column_names[0]
			label_column_name = (
				"ner_tags" if "ner_tags" in column_names else column_names[1]
			)

			if isinstance(features[label_column_name].feature, ClassLabel):
				label_list = features[label_column_name].feature.names
				# No need to convert the labels since they are already ints.
				label_to_id = {i: i for i in range(len(label_list))}
			else:
				label_list = get_label_list(dataset["train"][label_column_name])
				label_to_id = {l: i for i, l in enumerate(label_list)}
				
				# debug
				'''
				for i, l in enumerate(label_list):
					print(i, l)
				input('enter..')
				'''

		if task_name == 'ppi' or task_name == 'joint-ner-ppi':
			if do_lower_case:
				e1_start_id = tokenizer.convert_tokens_to_ids('[e1]')
				e2_start_id = tokenizer.convert_tokens_to_ids('[e2]')
				e1_end_id = tokenizer.convert_tokens_to_ids('[/e1]')
				e2_end_id = tokenizer.convert_tokens_to_ids('[/e2]')
			else:
				e1_start_id = tokenizer.convert_tokens_to_ids('[E1]')
				e2_start_id = tokenizer.convert_tokens_to_ids('[E2]')
				e1_end_id = tokenizer.convert_tokens_to_ids('[/E1]')
				e2_end_id = tokenizer.convert_tokens_to_ids('[/E2]')
			
			assert e1_start_id != e2_start_id != e1_end_id != e2_end_id != 1
			
		for phase, phase_dataset in dataset.items():
			
			''' debugging
			print('type(phase_dataset):', type(phase_dataset))
			import inspect
			print(inspect.getfullargspec(phase_dataset.map))
			'''
			if task_name == 'ner':
				features_dict[task_name][phase] = phase_dataset.map(
					convert_func_dict[task_name],
					fn_kwargs={'tokenizer': tokenizer, 
							   'text_column_name': text_column_name, 
							   'label_column_name': label_column_name, 
							   'label_to_id': label_to_id,
							   'label_all_tokens': data_args.label_all_tokens,
							   'padding': padding},
					batched=True,
					load_from_cache_file=False,
				)
			elif task_name == 'ppi':
				features_dict[task_name][phase] = phase_dataset.map(
					convert_func_dict[task_name],
					fn_kwargs={'tokenizer': tokenizer, 
							   'relation_representation': data_args.relation_representation,
							   'e1_start_id': e1_start_id,
							   'e2_start_id': e2_start_id,
							   'e1_end_id': e1_end_id,
							   'e2_end_id': e2_end_id,
							   'padding': padding},
					batched=True,
					load_from_cache_file=False,
				)
			elif task_name == 'joint-ner-ppi':
				
				
				
				# [START] temporarily loading ppi test set in dev path for PPI prediction. 05-02-2021
				if phase == 'validation':
					features_dict[task_name][phase] = phase_dataset.map(
						convert_func_dict['ppi'],
						fn_kwargs={'tokenizer': tokenizer, 
								   'relation_representation': data_args.relation_representation,
								   'e1_start_id': e1_start_id,
								   'e2_start_id': e2_start_id,
								   'e1_end_id': e1_end_id,
								   'e2_end_id': e2_end_id,
								   'padding': padding},
						batched=True,
						load_from_cache_file=False,
					)
				# [END] temporarily loading ppi test set in dev path for PPI prediction. 05-02-2021
				else:
					features_dict[task_name][phase] = phase_dataset.map(
						convert_func_dict[task_name],
						fn_kwargs={'tokenizer': tokenizer, 
								   'text_column_name': text_column_name, 
								   'label_column_name': label_column_name, 
								   'label_to_id': label_to_id,
								   'label_all_tokens': data_args.label_all_tokens,
								   'relation_representation': data_args.relation_representation,
								   'e1_start_id': e1_start_id,
								   'e2_start_id': e2_start_id,
								   'e1_end_id': e1_end_id,
								   'e2_end_id': e2_end_id,
								   'padding': padding,
								   'phase': phase},
						batched=True,
						load_from_cache_file=False,
					)
			#print(task_name, phase, len(phase_dataset), len(features_dict[task_name][phase]))
			
			# debugging
			"""
			if task_name == 'ppi' and phase == 'train':
			
				print('type(features_dict[task_name][phase]):', type(features_dict[task_name][phase])) # <class 'datasets.arrow_dataset.Dataset'>
				print('type(features_dict[task_name][phase][0]):', type(features_dict[task_name][phase][0])) # <class 'dict'>
				print(features_dict[task_name][phase][0])
				'''
				NER example
				{'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
				 'input_ids': [101, 1109, 8332, 170, 3530, 2285, 1177, 21794, 13378, 1104, 26181, 10947, 1162, 1110, 1227, 22997, 119, 102], 
				 'labels': [-100, 4, 4, 4, -100, -100, 4, -100, -100, 4, 4, -100, -100, 4, 4, -100, 4, -100], 
				 'ner': ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], 
				 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
				 'words': ['The', 'trace', 'aqueous', 'solubility', 'of', 'benzene', 'is', 'known36', '.']}
				'''
				#print(tokenizer)
				#print(tokenizer.do_lower_case)
				#print(tokenizer.decode(features_dict[task_name][phase][0]['input_ids']))
				#print(tokenizer.tokenize("The trace aqueous Gilchan Park solubility of benzene is known36."))
			"""					
			
			# [START] temporarily loading ppi test set in dev path for PPI prediction. 05-02-2021
			if task_name == 'joint-ner-ppi' and phase == 'validation':
				features_dict[task_name][phase].set_format(
					#type="torch",
					type=None,
					columns=columns_dict['ppi'],
				)
			# [END] temporarily loading ppi test set in dev path for PPI prediction. 05-02-2021
			else:		
				features_dict[task_name][phase].set_format(
					#type="torch",
					type=None,
					columns=columns_dict[task_name],
				)
				#print(task_name, phase, len(phase_dataset), len(features_dict[task_name][phase]))
				
			# debugging
			"""
			if task_name == 'ppi' and phase == 'train':
				print(features_dict[task_name][phase][0])
				'''
				NER example
				{'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
				 'input_ids': [101, 1109, 8332, 170, 3530, 2285, 1177, 21794, 13378, 1104, 26181, 10947, 1162, 1110, 1227, 22997, 119, 102], 
				 'labels': [-100, 4, 4, 4, -100, -100, 4, -100, -100, 4, 4, -100, -100, 4, 4, -100, 4, -100], 
				 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
				'''
			
			input('enter..')
			"""
	
	return features_dict

