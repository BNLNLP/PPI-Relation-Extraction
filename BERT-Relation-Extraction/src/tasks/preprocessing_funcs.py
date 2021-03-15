#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 18:12:22 2019

@author: weetee

modified by: Gilchan Park
-- Find the modifications by the tag [GP].
"""

import os
import re
import random
import copy
import pandas as pd
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from ..misc import save_as_pickle, load_pickle
from tqdm import tqdm
import logging
import csv
import re
import pickle

# [GP][START] - for cross-validation. 09-29-2020
from sklearn.model_selection import KFold, train_test_split
import numpy as np
# [GP][END] - for cross-validation. 09-29-2020

# [GP][START] - pre-processed PPI benchmark datasets (AImed, BioInfer, HPRD50, IEPA, LLL). 02-19-2021
from lxml import etree
# [GP][END] - pre-processed PPI benchmark datasets (AImed, BioInfer, HPRD50, IEPA, LLL). 02-19-2021


tqdm.pandas(desc="prog_bar")
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
					datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

def process_text(text, mode='train'):
	sents, relations, comments, blanks = [], [], [], []
	for i in range(int(len(text)/4)):
		sent = text[4*i]
		relation = text[4*i + 1]
		comment = text[4*i + 2]
		blank = text[4*i + 3]

		# check entries
		# [GP][START] - don't check index numbers since BioCreative may have different numbers. 11-27-2020
		'''
		if mode == 'train':
			assert int(re.match("^\d+", sent)[0]) == (i + 1)
		else:
			assert (int(re.match("^\d+", sent)[0]) - 8000) == (i + 1)
		'''
		# [GP][END] - don't check index numbers since BioCreative may have different numbers. 11-27-2020
		assert re.match("^Comment", comment)
		assert len(blank) == 1
		
		sent = re.findall("\"(.+)\"", sent)[0]
		sent = re.sub('<e1>', '[E1]', sent)
		sent = re.sub('</e1>', '[/E1]', sent)
		sent = re.sub('<e2>', '[E2]', sent)
		sent = re.sub('</e2>', '[/E2]', sent)
		sents.append(sent); relations.append(relation), comments.append(comment); blanks.append(blank)
	return sents, relations, comments, blanks


def preprocess_semeval2010_8(args):
	'''
	Data preprocessing for SemEval2010 task 8 dataset
	'''
	data_path = args.train_data #'./data/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT'
	logger.info("Reading training file %s..." % data_path)
	with open(data_path, 'r', encoding='utf8') as f:
		text = f.readlines()
	
	sents, relations, comments, blanks = process_text(text, 'train')
	df_train = pd.DataFrame(data={'sents': sents, 'relations': relations})
	
	data_path = args.test_data #'./data/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT'
	logger.info("Reading test file %s..." % data_path)
	with open(data_path, 'r', encoding='utf8') as f:
		text = f.readlines()
	
	sents, relations, comments, blanks = process_text(text, 'test')
	df_test = pd.DataFrame(data={'sents': sents, 'relations': relations})
	
	rm = Relations_Mapper(df_train['relations'])
	save_as_pickle('relations.pkl', rm)
	df_test['relations_id'] = df_test.progress_apply(lambda x: rm.rel2idx[x['relations']], axis=1)
	df_train['relations_id'] = df_train.progress_apply(lambda x: rm.rel2idx[x['relations']], axis=1)
	save_as_pickle('df_train.pkl', df_train)
	save_as_pickle('df_test.pkl', df_test)
	logger.info("Finished and saved!")
	
	return df_train, df_test, rm


# [GP][START] - preprocess BioCreative_BNL data. 12-17-2020

# this is a quick and very dirty string sanatizer for creating lookup keys based
# on the passage string
def modstring(s):
	return( s.replace(' ', '' ).replace('.','').upper()[0:100] )


# once a passage record is collected, this is invoked to print out one TSV line
def flush(sentences, selection, doc_num, text_num, pass_num, pass_total, annot, typ, notes, ian_comments):

	#print( "Executing flush operation" )
	passage = ""
	for s in sentences:
		#print( "    {0}".format( s ) )
		passage += s + ". "
	passage = passage[0:-2]    # remove extra .
	k = modstring( passage )
	#print( k )
	'''
	if k in string2row.keys():
		row = string2row.get( k )
		#print( "have row {0} for k".format( row ) )
	else:
		#print( "no row for k" )
		row = ''
	'''
	row = ''

	if selection != '':
		selected_sentence = sentences[int(selection)-1]
	else:
		selected_sentence = ''

	if typ == 'e':
		typ = 'enzyme'
	elif typ == 's':
		typ = 'structural'

	if doc_num == '':
		return

	#print( "\t".join( [row, doc_num, text_num, pass_num + "/" + pass_total, passage, selected_sentence, annot, typ, notes, ian_comments ] ) )

	return [row, doc_num, text_num, pass_num + "/" + pass_total, selected_sentence, annot, typ, notes]


# multi_flush handles multiple selected line numbers ( ie 1 & 2 )
def multi_flush(sentences, selections, doc_num, text_num, pass_num, pass_total, annot, typ, notes, ian_comments):
	for selection in re.split("\s*\&\s*", selections.strip()):
		return flush(sentences, selection, doc_num, text_num, pass_num, pass_total, annot, typ, notes, ian_comments)


def get_samples_from_bnl_annotation(file, num_of_sample=None, predefined_relation_type=None):
	passages = []
	
	state = 6
	sentence = 0 
	sentences = []
	doc_num = ''
	text_num = ''
	pass_num = ''
	pass_total = ''
	selection = ''
	annot = ''
	typ = ''
	notes = ''
	ian_comments = ''

	with open(file) as anf:
		for line in anf:
			line.strip()
			line = line.replace( '\r', '' )
			#print( "****line [{0}]\n".format( line ) )
			if line[0:3] == "doc":
				if ( state == 0 ):
					state = 1
					sentence_number = 1
					sentences = []
					m = re.match( "^doc\s+(\d+)\s+text\s+\#\s+(\d+)\s+passage\s+(\d+)/(\d+)", line )
					if m:
						#print( m.groups() )
						doc_num, text_num, pass_num, pass_total = m.groups();
					else:
						print( "bad doc line {0}".format( line ) )
						exit()
				else:
					print( "****hit doc state = {0} line {1}".format( state, line ) )
					exit()
			elif re.match( "(\d+)\)\s+(\S.*)", line ):
				m = re.match( "(\d+)\)\s+(\S.*)", line )
				sent_num, sent =  m.groups()
				sent_num = int( sent_num )
				#print( "{0}: [{1}]".format( sent_num, sent ) )
				if state != 1:
					print( "****hit sentence state = {0} line {1}".format( state, line ) )
					exit()
				if sent_num != sentence_number:
					print( "****hit sentence number = [{0}] != [{1}] line ]{2}]".format( sent_num, sentence_number, line ) )
					exit()
				sentences.append( sent )
				sentence_number += 1

			elif line[0:7].lower() == "select:":
				if ( state != 1 ):
					print( "****hit select state = {0} line {1}".format( state, line ) )
					exit()
				selection = line[7:].strip()
				state = 2

			elif line[0:6].lower() == "annot:":
				if ( state != 2 ):
					print( "****hit annot state = {0} line {1}".format( state, line ) )
					exit()
				annot = line[6:].strip()
				state = 3

			elif line[0:5].lower() == "type:":
				if ( state != 3 ):
					print( "****hit type state = {0} line {1}".format( state, line ) )
					exit()
				typ = line[5:].strip()
				state = 4

			elif line[0:6].lower() == "notes:":
				if ( state != 4 ):
					print( "****hit notes state = {0} line {1}".format( state, line ) )
					exit()
				notes = line[6:].strip()
				#print( "***notes = [{0}] line [{1}]\n".format( notes, line ) )
				state = 5

			elif line[0:2] == "I:":
				if ( state != 5 ):
					print( "****hit ian_comments state = {0} line [{1}]".format( state, line ) )
					exit()
				ian_comments = line[2:].strip()
				state = 6
			
			elif line[0:10] == "----------":
				if ( state < 5 ):
					print( "****hit flush state = {0} line {1}".format( state, line ) )
					exit()
				
				if typ != '':
					passages.append(multi_flush(sentences, selection, doc_num, text_num, pass_num, pass_total, annot, typ, notes, ian_comments))
					
				selection = 0
				doc_num = 0
				text_num = 0
				pass_num = 0
				pass_total = 0
				state = 0
				annot = ''
				typ = ''
				notes = ''
				ian_comments = ''

	# don't forget last record
	if typ != '':
		passages.append(multi_flush(sentences, selection, doc_num, text_num, pass_num, pass_total, annot, typ, notes, ian_comments))

	samples = {} # samples by documents

	for passage in passages:
		# passage -> [row, doc_num, text_num, pass_num + "/" + pass_total, selected_sentence, annot, typ, notes]
		doc_num = passage[1]
		unique_id = passage[1] + '_' + passage[2]
		sentence = passage[4].strip()
		relation = passage[5].strip()
		relation_type = passage[6].strip()
		comment = passage[7].strip()
		
		sentence = sentence.replace(u"\u00A0", " ") # remove non-breaking space. e.g., non-breaking space between 'alpha4' and 'integrins' in the row 9.
		
		if relation == '':
			continue
		
		if relation_type == '':
			print('relation_type is None!!')
			print(sentence)
			continue
			
		if relation_type not in ['structural', 'enzyme']: # exclude 'misc' for now since they are very few (only 3 as of 12-17-2020)
			print('this relation_type is undefined!!')
			print(sentence)
			continue
			
		relation = relation.split(';')
		rel_pairs = []
		for rel in relation:
			#print('rel:', rel.strip())
			
			entities = re.split(' -> | - | \? ', rel)
			entities = [x.strip() for x in entities]
			entities = [x.replace('_', ' ') for x in entities]
			
			#print('entities:', entities)
			
			if len(entities) != 2:
				print('this is not a pair relation:', entities)
				continue

			entity_grp_1 = [x.strip() for x in entities[0].split(',')] # e.g., FnBPA, FnBPB - fibronectin, fibrinogen, elastin
			entity_grp_2 = [x.strip() for x in entities[1].split(',')] # e.g., FnBPA, FnBPB - fibronectin, fibrinogen, elastin
			
			for e1 in entity_grp_1:
				for e2 in entity_grp_2:
					e1 = e1.replace('[', '').replace(']', '') # [ x ] indicates a family or class of proteins named x
					e2 = e2.replace('[', '').replace(']', '') # [ x ] indicates a family or class of proteins named x
					
					if e1 not in sentence or e2 not in sentence:
						print('not existence error - e1:', e1, '/ e2:', e2)
						continue
					
					if e1 == e2:
						print('e1 and e2 are the same - e1:', e1, '/ e2:', e2)
						continue
				
					tagged_sent = sentence.replace(e1, '<e1>' + e1 + '</e1>', 1)
					tagged_sent = tagged_sent.replace(e2, '<e2>' + e2 + '</e2>', 1)
					
					sample = []
					sample.append(unique_id + '\t"' + tagged_sent + '"')
					if predefined_relation_type != None:
						sample.append(predefined_relation_type + '(e1,e2)')
					else:
						sample.append(relation_type + '(e1,e2)')
					sample.append('Comment: ' + comment)
					sample.append('\n')
					
					
					# debugging
					'''
					print(unique_id + '\t"' + tagged_sent + '"')
					print(relation_type + '(e1,e2)')
					print('Comment: ' + comment)
					print('-------------------------------------------------\n')
					input('enter...')
					'''
					
					if doc_num in samples:
						samples[doc_num].append(sample)
					else:
						samples[doc_num] = [sample]
					
					if num_of_sample != None and len(samples) == num_of_sample:
						return samples

	return samples


def remove_unnecessary_token(psg_data):
	"""
	remove unnecessary tokens. '-' causes an error, so remove it.
	"""
	# debugging
	#debug_flag = False
	#print('Before:', psg_data)
	
	for tok_idx, tok in enumerate(psg_data['parsed_text']):
		if tok == '-':
			for x in range(len(psg_data['entities'])): # (ent_ncbi_id, ent_text, ent_type, ent_start, ent_end, offset, length)
				if tok_idx < psg_data['entities'][x][3]:
					psg_data['entities'][x][3] -= 1
					psg_data['entities'][x][4] -= 1
					
					#debug_flag = True
				elif tok_idx > psg_data['entities'][x][3] and tok_idx < psg_data['entities'][x][4]:
					psg_data['entities'][x][4] -= 1
					
					#debug_flag = True
	
	psg_data['parsed_text'] = [tok.replace('-', '') for tok in psg_data['parsed_text'] if tok != '-']		
	
	for x in range(len(psg_data['entities'])): # (ent_ncbi_id, ent_text, ent_type, ent_start, ent_end, offset, length)
		psg_data['entities'][x][1] = psg_data['entities'][x][1].replace('-', ' ')
		psg_data['entities'][x] = tuple(psg_data['entities'][x])
	
	# debugging
	#print('----------------------------------------------------')
	#print('After :', psg_data)
	#if debug_flag:
	#	input()


def get_samples_from_biocreative(file, num_of_sample=None, predefined_relation_type=None):
	"""
	Data preprocessing for original BioCreative PPI data ('PMtask_Relations_TrainingSet.json', 'PMtask_Relations_TestSet.json')
	
	TODO: Unlike other datasets, this processes the whole text rather than sentence units. If necessary, split texts into sentences using spacy or other tools.
	"""
	with open(file) as fp:
		data = json.load(fp)

	samples = {} # samples by documents
	
	for doc in data["documents"]:
		psg_data_list = []
		doc_id = doc["id"]
		for psg in doc["passages"]:
			psg_data = {}
			
			psg_text = psg["text"]
			psg_offset = psg["offset"]
			
			psg_data['raw_text'] = psg_text

			#psg_tokens = nlp(psg_text)
			
			'''
			# TODO: remove this after testing - consider only a single sentence passage.
			if len(list(psg_tokens.sents)) > 1:
				#print(psg_text)
				#print(list(psg_tokens.sents))
				continue
			'''
			
			entities = []
			for anno in psg["annotations"]:
				ent_text = anno["text"]
				ent_type = anno["infons"]["type"]
				if "NCBI GENE" in anno["infons"]:
					ent_id = anno["infons"]["NCBI GENE"]
				elif "identifier" in anno["infons"]:
					ent_id = anno["infons"]["identifier"]

				# debug
				if len(anno["infons"]) > 2:
					print('more than 3 values in infons')
					for i in anno["infons"]:
						print(i)
					input()
				  
				# TODO: handle entities constructed from several different tokens. 
				#       they are few (#21 in PMtask_Relations_TrainingSet.json), so ignore them for now.
				"""
				e.g., 
					tokens: ... 'nesprin-1', 'and', '-2', ...
					entity: 'nesprin-2'
						"locations": [
							{
							  "length": 7, 
							  "offset": 203
							}, 
							{
							  "length": 2, 
							  "offset": 217
							}
					
				"""
				if len(anno["locations"]) > 1:
					continue

				for loc in anno["locations"]:
					length = loc["length"]
					offset = loc["offset"] - psg_offset
					
					entity_annotation = [ent_id, ent_text, ent_type, offset, length]
					entities.append(entity_annotation)					
			
			psg_data['entities'] = entities
			
			#remove_unnecessary_token(psg_data) # remove '-' that causes an error in BERT Relation Extraction.

			psg_data_list.append(psg_data)
		
		def get_entity(gene_ncbi_id, entities):
			ret_val = []
			for entity in entities:
				if entity[0] == gene_ncbi_id:
					ret_val.append(entity)
			return ret_val

		for rel in doc["relations"]:
			rel_id = rel["id"]
			gene1_ncbi_id = rel["infons"]["Gene1"]
			gene2_ncbi_id = rel["infons"]["Gene2"]
			gene_rel = rel["infons"]["relation"]
			
			for psg_data in psg_data_list:
				gene1_entities = get_entity(gene1_ncbi_id, psg_data['entities'])
				gene2_entities = get_entity(gene2_ncbi_id, psg_data['entities'])
				
				for g1e in gene1_entities:
					for g2e in gene2_entities:

						g1_id = g1e[0]
						g1_text = g1e[1]
						g1_offset = g1e[3]
						g1_length = g1e[4]
						
						g2_id = g2e[0]
						g2_text = g2e[1]
						g2_offset = g2e[3]
						g2_length = g2e[4]
						
						# TODO: handle self PPIs. handle cases where a gene is inside the other gene.
						''' E.g., PMtask_Relation_TestSet.json: 
								  doc - "id": "7530509",
									  "infons": {
										"Gene1": "16590", 
										"Gene2": "16590", 
										"relation": "PPIm"
									  }, 
							E.g., gene 1: "leptin receptor", gene 2: "leptin" from PMtask_Relation_TestSet.json
							E.g., gene 1: "type 1 angiotensin II receptor", gene 2: "angiotensin II" from PMtask_Relation_TestSet.json
						'''
						if g1_id == g2_id \
						   or g1_offset == g2_offset \
						   or (g1_offset > g2_offset and g1_offset < (g2_offset + g2_length)) \
						   or (g2_offset > g1_offset and g2_offset < (g1_offset + g2_length)):
							continue
						
						
						# if text is too long, a programmatic error about tensor size occurs. 01-06-2021
						# TODO: find the exact maximum allowed length of text. 
						if len(psg_data['raw_text'].split()) > 40:
							continue
						
						
						tagged_text = psg_data['raw_text']
						
						if g1_offset < g2_offset: # replace first the one located in the rear.
							tagged_text = tagged_text[:g2_offset] + '<e2>' + g2_text + '</e2>' + tagged_text[g2_offset + g2_length:]
							tagged_text = tagged_text[:g1_offset] + '<e1>' + g1_text + '</e1>' + tagged_text[g1_offset + g1_length:]
						else:
							tagged_text = tagged_text[:g1_offset] + '<e1>' + g1_text + '</e1>' + tagged_text[g1_offset + g1_length:]
							tagged_text = tagged_text[:g2_offset] + '<e2>' + g2_text + '</e2>' + tagged_text[g2_offset + g2_length:]
							
						sample = []
						sample.append(doc_id + '_' + rel_id + '\t"' + tagged_text + '"')
						#sample.append(doc_id + '_' + rel_id + '\t"' + tagged_sentence + '"')
						sample.append(gene_rel + '(e1,e2)') # relation is always "PPIm"
						sample.append('Comment: ') # comment doesn't exist.
						sample.append('\n')

						# debugging
						'''
						if 'Functional properties of leptin receptor isoforms containing' in psg_data['raw_text']:
							print('g1_text:', g1_text)
							print('g2_text:', g2_text)
							print('tagged_text:', tagged_text)
							for k, v in psg_data.items():
								print(k, v)
							
							input('enter...')
						'''

						if doc_id in samples:
							samples[doc_id].append(sample)
						else:
							samples[doc_id] = [sample]
		
		''' debug
		print(samples)
		print('doc id:', doc["id"])
		input('enter..')
		'''
	
	return samples
# [GP][END] - preprocess BioCreative_BNL data. 12-17-2020


# [GP][START] - pre-processed PPI benchmark datasets (AImed, BioInfer, HPRD50, IEPA, LLL). 02-19-2021
def get_samples_from_ppi_benchmark(file, num_of_sample=None, predefined_relation_type=None):
	"""
	Data preprocessing for PPI benchmark datasets (AImed, BioInfer, HPRD50, IEPA, LLL).
	"""
	samples = {} # samples by documents
	
	xml_parser = etree.XMLParser(ns_clean=True)
	
	tree = etree.parse(file, xml_parser)
	root = tree.getroot()
			
	for doc_elem in root.findall('.//document'):
		doc_id = doc_elem.get('id')
		pubmed_id = doc_elem.get('origId')

		for sent_elem in doc_elem.findall('.//sentence'):
			sent_id = sent_elem.get('id')
			sent_txt = sent_elem.get('text')
			
			entities = {}
			for ent_elem in sent_elem.findall('.//entity'):
				ent_id = ent_elem.get('id')
				ent_seqId = ent_elem.get('seqId')
				ent_char_offset = ent_elem.get('charOffset')
				ent_text = ent_elem.get('text')
				ent_type = ent_elem.get('type')
				
				# TODO: handle entities that consist of separate words in a sentence. e.g., "65-70,82-86" -> 'muscle actin' in "...muscle and brain actin ..."
				if ',' in ent_char_offset:
					continue
					
				start_idx = int(ent_char_offset.split('-')[0])
				end_idx = int(ent_char_offset.split('-')[1]) + 1
				
				entities[ent_id] = {'start_idx': start_idx, 'end_idx': end_idx, 'text': ent_text}
				
				#assert ent_text == sent_txt[start_idx:end_idx]
				
				# debugging
				if ent_text != sent_txt[start_idx:end_idx]:
					print(ent_text)
					print(sent_txt[start_idx:end_idx])
					input('mismatch!!')

			for pair_elem in sent_elem.findall('.//pair'):
				pair_id = pair_elem.get('id')
				pair_e1 = pair_elem.get('e1')
				pair_e2 = pair_elem.get('e2')
				pair_interaction = pair_elem.get('interaction')
				
				# TODO: handle entities that consist of separate words in a sentence.
				if pair_e1 not in entities or pair_e2 not in entities:
					continue
				
				e1_start_idx = entities[pair_e1].get('start_idx')
				e2_start_idx = entities[pair_e2].get('start_idx')
				e1_end_idx = entities[pair_e1].get('end_idx')
				e2_end_idx = entities[pair_e2].get('end_idx')
				e1_text = entities[pair_e1].get('text')
				e2_text = entities[pair_e2].get('text')
				
				tagged_sent = sent_txt
				
				if e1_start_idx < e2_start_idx: # replace first the one located in the rear.
					tagged_sent = tagged_sent[:e2_start_idx] + '<e2>' + e2_text + '</e2>' + tagged_sent[e2_end_idx:]
					tagged_sent = tagged_sent[:e1_start_idx] + '<e1>' + e1_text + '</e1>' + tagged_sent[e1_end_idx:]
				else:
					tagged_sent = tagged_sent[:e1_start_idx] + '<e1>' + e1_text + '</e1>' + tagged_sent[e1_end_idx:]
					tagged_sent = tagged_sent[:e2_start_idx] + '<e2>' + e2_text + '</e2>' + tagged_sent[e2_end_idx:]
				
				relation_type = 'pos' if pair_interaction == 'True' else 'neg'

				sample = []
				sample.append(pair_id + '\t"' + tagged_sent + '"') # use pair_id for unique id.
				if predefined_relation_type != None:
					sample.append(predefined_relation_type + '(e1,e2)')
				else:
					sample.append(relation_type + '(e1,e2)')
				sample.append('Comment: ')
				sample.append('\n')
				
				if doc_id in samples:
					samples[doc_id].append(sample)
				else:
					samples[doc_id] = [sample]

	return samples
# [GP][END] - pre-processed PPI benchmark datasets (AImed, BioInfer, HPRD50, IEPA, LLL). 02-19-2021


# [GP][START] - PPI datasets pre-processing.
def store_data(dir, train, dev, test, idx=0):
	# flatten list
	train_text = [item for sublist in train for subsublist in sublist for item in subsublist] 
	dev_text = [item for sublist in dev for subsublist in sublist for item in subsublist] 
	test_text = [item for sublist in test for subsublist in sublist for item in subsublist]
	
	sents, relations, comments, blanks = process_text(train_text, 'train')
	df_train = pd.DataFrame(data={'sents': sents, 'relations': relations})

	sents, relations, comments, blanks = process_text(dev_text, 'dev')
	df_dev = pd.DataFrame(data={'sents': sents, 'relations': relations})

	sents, relations, comments, blanks = process_text(test_text, 'test')
	df_test = pd.DataFrame(data={'sents': sents, 'relations': relations})

	rm = Relations_Mapper(pd.concat([df_train['relations'], df_dev['relations'], df_test['relations']], axis=0))
	pickle.dump(rm, open(os.path.join(dir, 'relations_' + str(idx) + '.pkl'), "wb"))
	
	df_test['relations_id'] = df_test.progress_apply(lambda x: rm.rel2idx[x['relations']], axis=1)
	df_dev['relations_id'] = df_dev.progress_apply(lambda x: rm.rel2idx[x['relations']], axis=1)
	df_train['relations_id'] = df_train.progress_apply(lambda x: rm.rel2idx[x['relations']], axis=1)
	pickle.dump(df_train, open(os.path.join(dir, 'df_train_' + str(idx) + '.pkl'), "wb"))
	pickle.dump(df_dev, open(os.path.join(dir, 'df_dev_' + str(idx) + '.pkl'), "wb"))
	pickle.dump(df_test, open(os.path.join(dir, 'df_test_' + str(idx) + '.pkl'), "wb"))
	
	return df_train, df_dev, df_test, rm
	

def preprocess_ppi(args):
	"""
	Data preprocessing for PPI datasets.
	
	History:
		- pre-processed PPI annotations by BNL (Sean and Ian) from BioCreative datasets. 11-26-2020
		- pre-processed five PPI benchmark datasets: AImed, BioInfer, HPRD50, IEPA, LLL. 02-19-2021
	"""

	if args.do_cross_validation:
		# it used to retrieve a specific number of samples here, but since it reads a text from the beginning, it doesn't get random samples.
		# so, read the data all, and shuffle it and then get a specific number of samples. 12-23-2020
		# predefined_cls (if not set, it is None) is set when predefined lable is used instead of relation types from datasets. 01-06-2021
		if args.task == 'BioCreative':
			doc_samples = get_samples_from_biocreative(args.train_data, num_of_sample=None, predefined_relation_type=args.predefined_cls)
		elif args.task == 'BioCreative_BNL':
			doc_samples = get_samples_from_bnl_annotation(args.train_data, num_of_sample=None, predefined_relation_type=args.predefined_cls)
		else:
			doc_samples = get_samples_from_ppi_benchmark(args.train_data, num_of_sample=None, predefined_relation_type=args.predefined_cls)

		print('num of docs:', len(doc_samples))
		
		keys = list(doc_samples.keys())
		random.shuffle(keys)
		
		total_num = 0

		num_of_samples_for_eval = None if args.num_samples == -1 else args.num_samples
		samples = []
		counter = 0
		
		for k in keys:
			#print(k, '/ num of samples:', len(doc_samples[k]))
			total_num += len(doc_samples[k])
			
			if num_of_samples_for_eval != None:
				counter += len(doc_samples[k])
				if counter > num_of_samples_for_eval:
					max_idx = counter - num_of_samples_for_eval
					samples.append(doc_samples[k][:max_idx])
					break
				elif counter == num_of_samples_for_eval:
					samples.append(doc_samples[k])
					break
				else:
					samples.append(doc_samples[k])
			else:
				samples.append(doc_samples[k])
				
		print('num of total samples:', total_num)

		# debugging
		if num_of_samples_for_eval != None and num_of_samples_for_eval != len([item for sublist in samples for item in sublist]):
			input('sampling number is wrong!!')
		
		if num_of_samples_for_eval == None:
			dir = args.train_data.rsplit('/', 1)[0] + '/all'
		else:
			dir = args.train_data.rsplit('/', 1)[0] + '/' + str(args.num_samples)

		samples = np.array(samples)
		kfold = KFold(n_splits=args.num_of_folds, shuffle=False)
		for idx, (train_index, test_index) in enumerate(kfold.split(samples)):
			
			## Train/Validation(Dev)/Test split - 80/10/10, 70/15/15, 60/20/20 ratio
			if args.ratio == '80-10-10':
				dev_index = train_index[:(len(train_index)//9)]
				train_index = train_index[len(dev_index):]
			elif args.ratio == '70-15-15':
				test_index_adjusted = np.append(test_index, train_index[:(len(test_index)//2)])
				dev_index = train_index[(len(test_index)//2):(len(test_index)//2) + len(test_index_adjusted)]
				train_index = train_index[(len(test_index)//2) + len(dev_index):]
				test_index = test_index_adjusted
			elif args.ratio == '60-20-20':
				test_index_adjusted = np.append(test_index, train_index[:len(test_index)])
				dev_index = train_index[len(test_index):len(test_index) + len(test_index_adjusted)]
				train_index = train_index[len(test_index) + len(dev_index):]
				test_index = test_index_adjusted
				
			#print("TRAIN len:", len(train_index), "DEV len:", len(dev_index), "TEST len:", len(test_index))
			#print("TRAIN:", train_index, "DEV:", dev_index, "TEST:", test_index)

			train, dev, test = samples[train_index], samples[dev_index], samples[test_index]
			df_train, df_dev, df_test, rm = store_data(dir, train, dev, test, idx)

			if idx == 0:
				first_df_train = df_train
				first_df_dev = df_dev
				first_df_test = df_test
				first_rm = rm

			## Train/Test split
			# -> to use this code, 'samples' must be a list.
			'''
			print("TRAIN len:", len(train_index), "TEST len:", len(test_index))
			print("TRAIN:", train_index, "TEST:", test_index)
			
			train, test = samples[train_index], samples[test_index]

			train_text = [item for sublist in train for item in sublist] # flatten list
			test_text = [item for sublist in test for item in sublist] # flatten list
			
			sents, relations, comments, blanks = process_text(train_text, 'train')
			df_train = pd.DataFrame(data={'sents': sents, 'relations': relations})
			
			sents, relations, comments, blanks = process_text(test_text, 'test')
			df_test = pd.DataFrame(data={'sents': sents, 'relations': relations})

			rm = Relations_Mapper(pd.concat([df_train['relations'], df_test['relations']], axis=0))
			pickle.dump(rm, open(os.path.join(data_dir, 'relations_' + str(idx) + '.pkl'), "wb"))
			
			df_test['relations_id'] = df_test.progress_apply(lambda x: rm.rel2idx[x['relations']], axis=1)
			df_train['relations_id'] = df_train.progress_apply(lambda x: rm.rel2idx[x['relations']], axis=1)
			pickle.dump(df_train, open(os.path.join(data_dir, 'df_train_' + str(idx) + '.pkl'), "wb"))
			pickle.dump(df_test, open(os.path.join(data_dir, 'df_test_' + str(idx) + '.pkl'), "wb"))
			
			if idx == 0:
				first_df_train = df_train
				first_df_test = df_test
				first_rm = rm
			'''

		logger.info("Finished and saved!")
		
		#input('enter..')
		
		return first_df_train, first_df_dev, first_df_test, first_rm # return the first CV set.

	else:
		if args.task == 'BioCreative':
			train_samples = get_samples_from_biocreative(args.train_data, num_of_sample=None, predefined_relation_type=args.predefined_cls)
			if args.train_data != args.test_data:
				test_samples = get_samples_from_biocreative(args.test_data, num_of_sample=None, predefined_relation_type=args.predefined_cls)
		elif args.task == 'BioCreative_BNL':
			train_samples = get_samples_from_bnl_annotation(args.train_data, num_of_sample=None, predefined_relation_type=args.predefined_cls)
			if args.train_data != args.test_data:
				test_samples = get_samples_from_bnl_annotation(args.test_data, num_of_sample=None, predefined_relation_type=args.predefined_cls)
		else:
			train_samples = get_samples_from_ppi_benchmark(args.train_data, num_of_sample=None, predefined_relation_type=args.predefined_cls)
			if args.train_data != args.test_data:
				test_samples = get_samples_from_ppi_benchmark(args.test_data, num_of_sample=None, predefined_relation_type=args.predefined_cls)

		keys = list(train_samples.keys())
		random.shuffle(keys)
		train_samples = [train_samples[k] for k in keys]
		
		if args.train_data == args.test_data:
			## Train/Validation(Dev)/Test split - 80/10/10 ratio
			split_idx_1 = int(0.8 * len(train_samples))
			split_idx_2 = int(0.9 * len(train_samples))

			train, dev, test = train_samples[:split_idx_1], train_samples[split_idx_1:split_idx_2], train_samples[split_idx_2:]
		else:
			assert len(test_samples) != 0
			
			keys = list(test_samples.keys())
			random.shuffle(keys)
			test_samples = [test_samples[k] for k in keys]

			split_idx = len(test_samples)//3 # split data into dev and test data by the ratio 3:7
			
			train, dev, test = train_samples, test_samples[:split_idx], test_samples[split_idx:]
			
		# TODO: take into account the number of samples like the CV above.
		dir = args.train_data.rsplit('/', 1)[0] + '/all'
			
		df_train, df_dev, df_test, rm = store_data(dir, train, dev, test, 0)
		
		return df_train, df_dev, df_test, rm
# [GP][END] - PPI datasets pre-processing.


class Relations_Mapper(object):
	def __init__(self, relations):
		self.rel2idx = {}
		self.idx2rel = {}
		
		logger.info("Mapping relations to IDs...")
		self.n_classes = 0
		for relation in tqdm(relations):
			if relation not in self.rel2idx.keys():
				self.rel2idx[relation] = self.n_classes
				self.n_classes += 1
		
		for key, value in self.rel2idx.items():
			self.idx2rel[value] = key

class Pad_Sequence():
	"""
	collate_fn for dataloader to collate sequences of different lengths into a fixed length batch
	Returns padded x sequence, y sequence, x lengths and y lengths of batch
	"""
	def __init__(self, seq_pad_value, label_pad_value=-1, label2_pad_value=-1,\
				 ):
		self.seq_pad_value = seq_pad_value
		self.label_pad_value = label_pad_value
		self.label2_pad_value = label2_pad_value
		
	def __call__(self, batch):
		sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
		seqs = [x[0] for x in sorted_batch]
		seqs_padded = pad_sequence(seqs, batch_first=True, padding_value=self.seq_pad_value)
		x_lengths = torch.LongTensor([len(x) for x in seqs])
		
		labels = list(map(lambda x: x[1], sorted_batch))
		labels_padded = pad_sequence(labels, batch_first=True, padding_value=self.label_pad_value)
		y_lengths = torch.LongTensor([len(x) for x in labels])
		
		labels2 = list(map(lambda x: x[2], sorted_batch))
		labels2_padded = pad_sequence(labels2, batch_first=True, padding_value=self.label2_pad_value)
		y2_lengths = torch.LongTensor([len(x) for x in labels2])
		
		return seqs_padded, labels_padded, labels2_padded, \
				x_lengths, y_lengths, y2_lengths

def get_e1e2_start(tokenizer, x, e1_id, e2_id):
	try:
		e1_e2_start = ([i for i, e in enumerate(x) if e == e1_id][0],\
						[i for i, e in enumerate(x) if e == e2_id][0])
	except Exception as e:
		e1_e2_start = None
		print(e)
	
	# [GP][START]
	# debugging
	'''
	print('e1_id:', e1_id, '/ e2_id:', e2_id)
	print('e1_e2_start:', e1_e2_start)
	print(x)
	print(' '.join([tokenizer.convert_ids_to_tokens(t) for t in x]))
	if e1_e2_start is None:
		print('ERRORRRRRRRRRRRRRRRRRRRRRRRR!!!!')
		input('enter...')
	'''
	# [GP][END]

	return e1_e2_start

class semeval_dataset(Dataset):
	def __init__(self, df, tokenizer, e1_id, e2_id):
		self.e1_id = e1_id
		self.e2_id = e2_id
		self.df = df
		logger.info("Tokenizing data...")
		self.df['input'] = self.df.progress_apply(lambda x: tokenizer.encode(x['sents']),\
															 axis=1)
		
		self.df['e1_e2_start'] = self.df.progress_apply(lambda x: get_e1e2_start(tokenizer, x['input'],\
													   e1_id=self.e1_id, e2_id=self.e2_id), axis=1)
		print("\nInvalid rows/total: %d/%d" % (df['e1_e2_start'].isnull().sum(), len(df)))
		self.df.dropna(axis=0, inplace=True)
	
	def __len__(self,):
		return len(self.df)
		
	def __getitem__(self, idx):
		return torch.LongTensor(self.df.iloc[idx]['input']),\
				torch.LongTensor(self.df.iloc[idx]['e1_e2_start']),\
				torch.LongTensor([self.df.iloc[idx]['relations_id']])

def preprocess_fewrel(args, do_lower_case=True):
	'''
	train: train_wiki.json
	test: val_wiki.json
	For 5 way 1 shot
	'''
	def process_data(data_dict):
		sents = []
		labels = []
		for relation, dataset in data_dict.items():
			for data in dataset:
				# first, get & verify the positions of entities
				h_pos, t_pos = data['h'][-1], data['t'][-1]
				
				if not len(h_pos) == len(t_pos) == 1: # remove one-to-many relation mappings
					continue
				
				h_pos, t_pos = h_pos[0], t_pos[0]
				
				if len(h_pos) > 1:
					running_list = [i for i in range(min(h_pos), max(h_pos) + 1)]
					assert h_pos == running_list
					h_pos = [h_pos[0], h_pos[-1] + 1]
				else:
					h_pos.append(h_pos[0] + 1)
				
				if len(t_pos) > 1:
					running_list = [i for i in range(min(t_pos), max(t_pos) + 1)]
					assert t_pos == running_list
					t_pos = [t_pos[0], t_pos[-1] + 1]
				else:
					t_pos.append(t_pos[0] + 1)
				
				if (t_pos[0] <= h_pos[-1] <= t_pos[-1]) or (h_pos[0] <= t_pos[-1] <= h_pos[-1]): # remove entities not separated by at least one token 
					continue
				
				if do_lower_case:
					data['tokens'] = [token.lower() for token in data['tokens']]
				
				# add entity markers
				if h_pos[-1] < t_pos[0]:
					tokens = data['tokens'][:h_pos[0]] + ['[E1]'] + data['tokens'][h_pos[0]:h_pos[1]] \
							+ ['[/E1]'] + data['tokens'][h_pos[1]:t_pos[0]] + ['[E2]'] + \
							data['tokens'][t_pos[0]:t_pos[1]] + ['[/E2]'] + data['tokens'][t_pos[1]:]
				else:
					tokens = data['tokens'][:t_pos[0]] + ['[E2]'] + data['tokens'][t_pos[0]:t_pos[1]] \
							+ ['[/E2]'] + data['tokens'][t_pos[1]:h_pos[0]] + ['[E1]'] + \
							data['tokens'][h_pos[0]:h_pos[1]] + ['[/E1]'] + data['tokens'][h_pos[1]:]
				
				assert len(tokens) == (len(data['tokens']) + 4)
				sents.append(tokens)
				labels.append(relation)
		return sents, labels
		
	with open('./data/fewrel/train_wiki.json') as f:
		train_data = json.load(f)
		
	with  open('./data/fewrel/val_wiki.json') as f:
		test_data = json.load(f)
	
	train_sents, train_labels = process_data(train_data)
	test_sents, test_labels = process_data(test_data)
	
	df_train = pd.DataFrame(data={'sents': train_sents, 'labels': train_labels})
	df_test = pd.DataFrame(data={'sents': test_sents, 'labels': test_labels})
	
	rm = Relations_Mapper(list(df_train['labels'].unique()))
	save_as_pickle('relations.pkl', rm)
	df_train['labels'] = df_train.progress_apply(lambda x: rm.rel2idx[x['labels']], axis=1)
	
	return df_train, df_test


# [GP][START] - added dataset number parameter.
def load_dataloaders(args, dataset_num):
# [GP][END] - added dataset number parameter.
	if args.model_no == 0:
		from ..model.BERT.tokenization_bert import BertTokenizer as Tokenizer
		model = args.model_size#'bert-large-uncased' 'bert-base-uncased'
		lower_case = True
		model_name = 'BERT'
	elif args.model_no == 1:
		from ..model.ALBERT.tokenization_albert import AlbertTokenizer as Tokenizer
		model = args.model_size #'albert-base-v2'
		lower_case = True
		model_name = 'ALBERT'
	elif args.model_no == 2:
		from ..model.BERT.tokenization_bert import BertTokenizer as Tokenizer
		model = 'bert-base-uncased'
		lower_case = False
		model_name = 'BioBERT'
		
	if os.path.isfile("./data/%s_tokenizer.pkl" % model_name):
		tokenizer = load_pickle("%s_tokenizer.pkl" % model_name)
		logger.info("Loaded tokenizer from pre-trained blanks model")
	else:
		logger.info("Pre-trained blanks tokenizer not found, initializing new tokenizer...")
		if args.model_no == 2:
			tokenizer = Tokenizer(vocab_file='./additional_models/biobert_v1.1_pubmed/vocab.txt',
								  do_lower_case=False)
		else:
			tokenizer = Tokenizer.from_pretrained(model, do_lower_case=False)
		tokenizer.add_tokens(['[E1]', '[/E1]', '[E2]', '[/E2]', '[BLANK]'])

		save_as_pickle("%s_tokenizer.pkl" % model_name, tokenizer)
		logger.info("Saved %s tokenizer at ./data/%s_tokenizer.pkl" %(model_name, model_name))
	
	e1_id = tokenizer.convert_tokens_to_ids('[E1]')
	e2_id = tokenizer.convert_tokens_to_ids('[E2]')
	assert e1_id != e2_id != 1
	
	if args.task == 'fewrel':
		df_train, df_test = preprocess_fewrel(args, do_lower_case=lower_case)
		train_loader = fewrel_dataset(df_train, tokenizer=tokenizer, seq_pad_value=tokenizer.pad_token_id,
									  e1_id=e1_id, e2_id=e2_id)
		train_length = len(train_loader)
		test_loader, test_length = None, None
	else:
		if args.task == 'semeval':
			relations_path = './data/relations.pkl'
			train_path = './data/df_train.pkl'
			test_path = './data/df_test.pkl'
			if os.path.isfile(relations_path) and os.path.isfile(train_path) and os.path.isfile(test_path):
				rm = load_pickle('relations.pkl')
				df_train = load_pickle('df_train.pkl')
				df_test = load_pickle('df_test.pkl')
				logger.info("Loaded preproccessed data.")
			else:
				df_train, df_test, rm = preprocess_semeval2010_8(args)
		else:
			# [GP][START] - PPI datasets pre-processing.
			#			  - pre-processed BioCreative_BNL data. 11-26-2020
			# 			  - added dev set. 12-23-2020
			#			  - pre-processed AImed, BioInfer, HPRD50, IEPA, LLL. 02-19-2021
			if args.num_samples == -1:
				data_dir = args.train_data.rsplit('/', 1)[0] + '/all'
			else:
				data_dir = args.train_data.rsplit('/', 1)[0] + '/' + str(args.num_samples)
	
			if not os.path.exists(data_dir):
				os.makedirs(data_dir)
			
			relations_path = os.path.join(data_dir, 'relations_' + str(dataset_num) + '.pkl')
			train_path = os.path.join(data_dir, 'df_train_' + str(dataset_num) + '.pkl')
			dev_path = os.path.join(data_dir, 'df_dev_' + str(dataset_num) + '.pkl')
			test_path = os.path.join(data_dir, 'df_test_' + str(dataset_num) + '.pkl')

			if os.path.isfile(relations_path) and os.path.isfile(train_path) and os.path.isfile(dev_path) and os.path.isfile(test_path):
				rm = pickle.load(open(relations_path, "rb"))
				df_train = pickle.load(open(train_path, "rb"))
				df_dev = pickle.load(open(dev_path, "rb"))
				df_test = pickle.load(open(test_path, "rb"))
				logger.info("Loaded preproccessed data.")
			else:
				df_train, df_dev, df_test, rm = preprocess_ppi(args)
			# [GP][END] - PPI datasets pre-processing.

		train_set = semeval_dataset(df_train, tokenizer=tokenizer, e1_id=e1_id, e2_id=e2_id)
		test_set = semeval_dataset(df_test, tokenizer=tokenizer, e1_id=e1_id, e2_id=e2_id)
		train_length = len(train_set); test_length = len(test_set)
		# [GP][START] - added dev set. 12-23-2020
		dev_set = semeval_dataset(df_dev, tokenizer=tokenizer, e1_id=e1_id, e2_id=e2_id)
		dev_length = len(dev_set)
		# [GP][END] - added dev set. 12-23-2020
		PS = Pad_Sequence(seq_pad_value=tokenizer.pad_token_id,\
						  label_pad_value=tokenizer.pad_token_id,\
						  label2_pad_value=-1)
		train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, \
								  num_workers=0, collate_fn=PS, pin_memory=False)
		test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, \
								  num_workers=0, collate_fn=PS, pin_memory=False)
		# [GP][START] - added dev set. 12-23-2020
		dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=True, \
								num_workers=0, collate_fn=PS, pin_memory=False)
		# [GP][END] - added dev set. 12-23-2020
	
	# [GP][START] - added dev set. 12-23-2020	
	return train_loader, dev_loader, test_loader, train_length, dev_length, test_length
	# [GP][END] - added dev set. 12-23-2020


class fewrel_dataset(Dataset):
	def __init__(self, df, tokenizer, seq_pad_value, e1_id, e2_id):
		self.e1_id = e1_id
		self.e2_id = e2_id
		self.N = 5
		self.K = 1
		self.df = df
		
		logger.info("Tokenizing data...")
		self.df['sents'] = self.df.progress_apply(lambda x: tokenizer.encode(" ".join(x['sents'])),\
									  axis=1)
		self.df['e1_e2_start'] = self.df.progress_apply(lambda x: get_e1e2_start(x['sents'],\
													   e1_id=self.e1_id, e2_id=self.e2_id), axis=1)
		print("\nInvalid rows/total: %d/%d" % (self.df['e1_e2_start'].isnull().sum(), len(self.df)))
		self.df.dropna(axis=0, inplace=True)
		
		self.relations = list(self.df['labels'].unique())
		
		self.seq_pad_value = seq_pad_value
			
	def __len__(self,):
		return len(self.df)
	
	def __getitem__(self, idx):
		target_relation = self.df['labels'].iloc[idx]
		relations_pool = copy.deepcopy(self.relations)
		relations_pool.remove(target_relation)
		sampled_relation = random.sample(relations_pool, self.N - 1)
		sampled_relation.append(target_relation)
		
		target_idx = self.N - 1
	
		e1_e2_start = []
		meta_train_input, meta_train_labels = [], []
		for sample_idx, r in enumerate(sampled_relation):
			filtered_samples = self.df[self.df['labels'] == r][['sents', 'e1_e2_start', 'labels']]
			sampled_idxs = random.sample(list(i for i in range(len(filtered_samples))), self.K)
			
			sampled_sents, sampled_e1_e2_starts = [], []
			for sampled_idx in sampled_idxs:
				sampled_sent = filtered_samples['sents'].iloc[sampled_idx]
				sampled_e1_e2_start = filtered_samples['e1_e2_start'].iloc[sampled_idx]
				
				assert filtered_samples['labels'].iloc[sampled_idx] == r
				
				sampled_sents.append(sampled_sent)
				sampled_e1_e2_starts.append(sampled_e1_e2_start)
			
			meta_train_input.append(torch.LongTensor(sampled_sents).squeeze())
			e1_e2_start.append(sampled_e1_e2_starts[0])
			
			meta_train_labels.append([sample_idx])
			
		meta_test_input = self.df['sents'].iloc[idx]
		meta_test_labels = [target_idx]
		
		e1_e2_start.append(get_e1e2_start(meta_test_input, e1_id=self.e1_id, e2_id=self.e2_id))
		e1_e2_start = torch.LongTensor(e1_e2_start).squeeze()
		
		meta_input = meta_train_input + [torch.LongTensor(meta_test_input)]
		meta_labels = meta_train_labels + [meta_test_labels]
		meta_input_padded = pad_sequence(meta_input, batch_first=True, padding_value=self.seq_pad_value).squeeze()
		return meta_input_padded, e1_e2_start, torch.LongTensor(meta_labels).squeeze()
