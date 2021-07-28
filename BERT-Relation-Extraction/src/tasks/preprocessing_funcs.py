#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 18:12:22 2019

@author: weetee

modified by: Gilchan Park
-- Find the modifications by the tag [GP].
"""

import os
import sys
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
		
		comment = re.sub('Comment:\s*', '', comment)

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


# [GP][START] - BioCreative_type data pre-processing. 12-17-2020

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

	return [row, doc_num, text_num, selection, pass_num + "/" + pass_total, selected_sentence, annot, typ, notes]


# multi_flush handles multiple selected line numbers ( ie 1 & 2 )
def multi_flush(sentences, selections, doc_num, text_num, pass_num, pass_total, annot, typ, notes, ian_comments):
	for selection in re.split("\s*\&\s*", selections.strip()):
		return flush(sentences, selection, doc_num, text_num, pass_num, pass_total, annot, typ, notes, ian_comments)


def get_samples_from_biocreative_type(file, num_of_sample=None, predefined_relation_type=None):
	"""
	Data preprocessing for BioCreative Typed data (annotated by BNL)
	
	BioCreative data only indicates whether two proteins interact or not, and it doesn't have types of relation information.
	BioCreative_type data has types of relations such as 'enzyme', 'structural'.
	"""
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
	unique_sents = set() # used to remove duplicates

	for passage in passages:
		# passage -> [row, doc_num, text_num, sentence_num, pass_num + "/" + pass_total, selected_sentence, annot, typ, notes]

		doc_num = passage[1]
		unique_id = passage[1] + '_' + passage[2] + '_' + passage[3]
		sentence = passage[5].strip()
		relation = passage[6].strip()
		relation_type = passage[7].strip()
		comment = passage[8].strip()
		
		sentence = sentence.replace(u"\u00A0", " ") # remove non-breaking space. e.g., non-breaking space between 'alpha4' and 'integrins' in the row 9.
		
		if relation == '':
			continue
		
		if relation_type == '':
			logger.info('relation_type is None: %s', sentence)
			continue
			
		if relation_type not in ['structural', 'enzyme']: # exclude 'misc' for now since they are very few (only 3 as of 12-17-2020)
			logger.info('this relation_type is undefined: %s', sentence)
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
				logger.info('this is not a pair relation: %s', ' '.join(entities))
				continue

			entity_grp_1 = [x.strip() for x in entities[0].split(',')] # e.g., FnBPA, FnBPB - fibronectin, fibrinogen, elastin
			entity_grp_2 = [x.strip() for x in entities[1].split(',')] # e.g., FnBPA, FnBPB - fibronectin, fibrinogen, elastin
			
			for e1 in entity_grp_1:
				for e2 in entity_grp_2:
					# [ x ] indicates a family or class of proteins named x
					e1 = re.sub(r'\[\s*', '', e1)
					e1 = re.sub(r'\s*\]', '', e1)
					e2 = re.sub(r'\[\s*', '', e2)
					e2 = re.sub(r'\s*\]', '', e2)
					
					if e1 not in sentence or e2 not in sentence:
						logger.info('not existence error - e1: %s / e2: %s', e1, e2)
						continue
					
					if e1 == e2:
						logger.info('e1 and e2 are the same - e1: %s / e2: %s', e1, e2)
						continue
					
					# TODO: fix this problem! this sentence is the only replacement error in the data. 04-29-2021
					'''
					e.g., 
					sentence: Yeast epiarginase regulation, an enzyme-enzyme activity control: identification of residues of ornithine carbamoyltransferase and arginase responsible for enzyme catalytic and regulatory activities.
					tagged_sent: Yeast epi<e2>arginase</e2> regulation, an enzyme-enzyme activity control: identification of residues of <e1>ornithine carbamoyltransferase</e1> and arginase responsible for enzyme catalytic and regulatory activities.
					e1: ornithine carbamoyltransferase / e2: arginase
					'''
					tagged_sent = sentence.replace(e1, '<e1>' + e1 + '</e1>', 1)
					if sentence.startswith('Yeast epiarginase regulation,'):
						tagged_sent = tagged_sent.replace('and arginase responsible', 'and <e2>arginase</e2> responsible', 1)
					else:
						tagged_sent = tagged_sent.replace(e2, '<e2>' + e2 + '</e2>', 1)
					# debug
					'''
					if (tagged_sent.index('<e1>')-1 >= 0 and bool(re.match('^[a-zA-Z0-9]+$', tagged_sent[tagged_sent.index('<e1>')-1]))) or \
					   (tagged_sent.index('<e2>')-1 >= 0 and bool(re.match('^[a-zA-Z0-9]+$', tagged_sent[tagged_sent.index('<e2>')-1]))):
						print('sentence:', sentence)
						print('tagged_sent:', tagged_sent)
						print('e1:', e1, '/ e2:', e2)
						print(tagged_sent[tagged_sent.index('<e1>')-1])
						print(tagged_sent[tagged_sent.index('<e2>')-1])
						input('enter...')
					'''

					# TODO: handle the cases where entities are overlapped.
					e1_s_idx = tagged_sent.index('<e1>')
					e1_e_idx = tagged_sent.index('</e1>')
					e2_s_idx = tagged_sent.index('<e2>')
					e2_e_idx = tagged_sent.index('</e2>')
					
					if (e1_s_idx > e2_s_idx and e1_s_idx < e2_e_idx) or (e2_s_idx > e1_s_idx and e2_s_idx < e1_e_idx):
						logger.info('entities are overlapped: %s', tagged_sent)
						continue
						
					if tagged_sent in unique_sents: # all sentences with tags must be unique. this is just in case.
						'''
						duplicate error example: TODO: deal with directional relation. For now, '->' and '-' are treated equally.
							doc 283 text # 568  passage 2/2:
							annot:  Relaxin-3 -> RXFP3, RXFP4, RXFP1; Relaxin-3 - RXFP3, RXFP4, RXFP1
						'''
						logger.info('duplicate sent: %s', tagged_sent)
						continue
					else:
						unique_sents.add(tagged_sent)
					
					sample = []
					sample.append(unique_id + '\t"' + tagged_sent + '"')
					if predefined_relation_type != None:
						sample.append(predefined_relation_type + '(e1,e2)')
					else:
						sample.append(relation_type + '(e1,e2)')
					#sample.append('Comment: ' + comment)
					sample.append('Comment: ' + unique_id) # added sentence ids to be used to group the same sentences for different pairs in joint ner ppi learning. 05-01-2021
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
	unique_texts = set() # used to remove duplicates
	
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
						
						# TODO: handle self PPIs and cases where entities are overlapped.
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
						
						if tagged_text in unique_texts:
							print('tagged_text sent:', tagged_text)
							continue
						else:
							unique_texts.add(tagged_text)
							
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
# [GP][END] - BioCreative_type data pre-processing. 12-17-2020


# [GP][START] - PPI benchmark datasets pre-processing. 02-19-2021
def get_samples_from_ppi_benchmark(file, num_of_sample=None, predefined_relation_type=None):
	"""
	Data preprocessing for PPI benchmark datasets (AImed, BioInfer, HPRD50, IEPA, LLL).
	"""
	samples = {} # samples by documents
	unique_sents = set() # used to remove duplicates
	
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
				
				
				# TODO: handle self PPIs and cases where entities are overlapped.
				''' E.g., AImed.xml: 
						<sentence id="AIMed.d198.s1687" seqId="s1687" text="Mutagenesis of the erythropoietin receptor (EPOR) permits analysis of the contribution that individual amino acid residues make to erythropoietin (EPO) binding.">
							<entity charOffset="19-32" id="AIMed.d198.s1687.e0" seqId="e5485" text="erythropoietin" type="protein" />
							<entity charOffset="19-41" id="AIMed.d198.s1687.e1" seqId="e5487" text="erythropoietin receptor" type="protein" />
				'''
				''' debugging
				if pair_e1 == pair_e2:
					print('pair_id:', pair_id)
					print('pair_e1 == pair_e2:', pair_e1, pair_e2)
					input('enter..')
					
				if e1_start_idx == e2_start_idx:
					print('pair_id:', pair_id)
					print('e1_start_idx == e2_start_idx:', e1_start_idx, e2_start_idx)
					input('enter..')
				
				if e1_start_idx > e2_start_idx and e1_start_idx < e2_end_idx:
					print('pair_id:', pair_id)
					print('e1_start_idx > e2_start_idx and e1_start_idx < e2_end_idx:', e1_start_idx, e2_start_idx, e2_end_idx)
					input('enter..')
				
				if e2_start_idx > e1_start_idx and e2_start_idx < e1_end_idx:
					print('pair_id:', pair_id)
					print('e2_start_idx > e1_start_idx and e2_start_idx < e1_end_idx:', e2_start_idx, e1_start_idx, e1_end_idx)
					input('enter..')
				'''
				if pair_e1 == pair_e2 \
				   or e1_start_idx == e2_start_idx \
				   or (e1_start_idx > e2_start_idx and e1_start_idx < e2_end_idx) \
				   or (e2_start_idx > e1_start_idx and e2_start_idx < e1_end_idx):
					continue
				
				tagged_sent = sent_txt

				if e1_start_idx < e2_start_idx: # replace first the one located in the rear.
					tagged_sent = tagged_sent[:e2_start_idx] + '<e2>' + e2_text + '</e2>' + tagged_sent[e2_end_idx:]
					tagged_sent = tagged_sent[:e1_start_idx] + '<e1>' + e1_text + '</e1>' + tagged_sent[e1_end_idx:]
				else:
					tagged_sent = tagged_sent[:e1_start_idx] + '<e1>' + e1_text + '</e1>' + tagged_sent[e1_end_idx:]
					tagged_sent = tagged_sent[:e2_start_idx] + '<e2>' + e2_text + '</e2>' + tagged_sent[e2_end_idx:]
				
				if tagged_sent in unique_sents: # all sentences with tags must be unique. this is for just in case.
					print('duplicate sent:', tagged_sent)
					continue
				else:
					unique_sents.add(tagged_sent)

				relation_type = 'positive' if pair_interaction == 'True' else 'negative'

				sample = []
				sample.append(pair_id + '\t"' + tagged_sent + '"') # use pair_id for unique id.
				if predefined_relation_type != None:
					sample.append(predefined_relation_type + '(e1,e2)')
				else:
					sample.append(relation_type + '(e1,e2)')
				#sample.append('Comment: ')
				sample.append('Comment: ' + sent_id) # added sentence ids to be used to group the same sentences for different pairs in joint ner ppi learning. 05-01-2021
				sample.append('\n')

				if doc_id in samples:
					samples[doc_id].append(sample)
				else:
					samples[doc_id] = [sample]

	return samples
# [GP][END] - PPI benchmark datasets pre-processing. 02-19-2021


# [GP][START] - PPI benchmark typed datasets pre-processing. 03-18-2021
def get_samples_from_ppi_benchmark_type(file, num_of_sample=None, predefined_relation_type=None):
	"""
	Data preprocessing for PPI benchmark datasets typed (annotated by BNL).
	
	This data contains types of relations such as 'enzyme', 'structural'.
	"""
	
	# First, read samples from the original file, and then replace positive relation_type with new types (e.g., enzyme, structural) 
	# e.g., orig_file = PPI/original/AImed/AImed.xml
	#		type_file = PPI/type_annotation/AImed_type/aimed_type_annotations_srm.tsv
	original_file_dir = file.rsplit('/', 1)[0].rsplit('_', 1)[0] + '/'
	original_file_dir = original_file_dir.replace('type_annotation', 'original')
	
	original_file = None
	for f in os.listdir(original_file_dir):
		if f.endswith(".xml"):
			original_file = os.path.join(original_file_dir, f)
			break
	
	if original_file == None:
		logger.error("original file does not exist!!")
		sys.exit()

	samples = get_samples_from_ppi_benchmark(original_file)

	def update_rel_type(samples, pair_id, relation_type):
		for doc_id, sample_list in samples.items():
			for idx, sample in enumerate(sample_list):
				sample_pair_id = sample[0].split('\t', 1)[0]
				if pair_id == sample_pair_id:
					samples[doc_id][idx][1] = relation_type + '(e1,e2)'
					return
		return
	
	# [START] used for negative_annotation_files e.g., passed_full_aimed_training.tsv 05-04-2021
	neg_sample_processing = False
	
	if neg_sample_processing is True:
		negative_pair_ids = []
	# [END] used for negative_annotation_files e.g., passed_full_aimed_training.tsv 05-04-2021
	
	with open(file) as fp:
		reader = csv.reader(fp, delimiter="\t")
		for row in reader:
			if not row:
				continue # skip empty lines

			if not row[0].startswith('sentence'):
				pair_id = row[0].strip()
				relation_type = row[1].strip()
				
				if relation_type in ['enzyme', 'structural']:
					update_rel_type(samples, pair_id, relation_type)
				
				# [START] used for negative_annotation_files e.g., passed_full_aimed_training.tsv 05-04-2021
				if neg_sample_processing is True:
					if relation_type == 'negative':
						negative_pair_ids.append(pair_id)	
				# [END] used for negative_annotation_files e.g., passed_full_aimed_training.tsv 05-04-2021
				
				# debugging
				if relation_type not in ['enzyme', 'structural', 'negative', '?']:
					print('unknown relation type:', relation_type)
					input('enter...')
	
	# [START] used for negative_annotation_files e.g., passed_full_aimed_training.tsv 05-04-2021
	if neg_sample_processing is True:
		negative_samples = {}
		for doc_id, sample_list in samples.items():
			for idx, sample in enumerate(sample_list):
				sample_pair_id = sample[0].split('\t', 1)[0]
				if sample_pair_id in negative_pair_ids:
					if doc_id in negative_samples:
						negative_samples[doc_id].append(sample)
					else:
						negative_samples[doc_id] = [sample]
		
		samples = negative_samples
	# [END] used for negative_annotation_files e.g., passed_full_aimed_training.tsv 05-04-2021
	
	# remove samples tagged with 'positive' since the classes are 'enzyme', 'structural', 'negative'.
	for doc_id, sample_list in samples.items():
		# [START] used for negative_annotation_files e.g., passed_full_aimed_training.tsv 05-04-2021
		if neg_sample_processing is True:
			samples[doc_id] = [x for x in sample_list if not x[1].startswith('positive')]
		# [END] used for negative_annotation_files e.g., passed_full_aimed_training.tsv 05-04-2021
		else:
			# [START] use this option to remove 'negative' samples from original data. 05-04-2021
			samples[doc_id] = [x for x in sample_list if not x[1].startswith('positive') and not x[1].startswith('negative')]
			# [END] use this option to remove 'negative' samples from original data. 05-04-2021
			
	return samples
# [GP][END] - PPI benchmark typed datasets pre-processing. 03-18-2021


# [GP][START] - PPI datasets pre-processing.
def store_data(dir, train, dev, test, classes, idx=0):
	# flatten list
	train_text = [item for sublist in train for subsublist in sublist for item in subsublist] 
	test_text = [item for sublist in test for subsublist in sublist for item in subsublist]
	if dev is not None:
		dev_text = [item for sublist in dev for subsublist in sublist for item in subsublist] 
		
	sents, relations, comments, blanks = process_text(train_text, 'train') # comments are used to store sentence ids. 05-01-2021
	df_train = pd.DataFrame(data={'sents': sents, 'relations': relations, 'sent_ids': comments}) # added sentence ids to be used to group the same sentences for different pairs in joint ner ppi learning. 05-01-2021
	sents, relations, comments, blanks = process_text(test_text, 'test')
	df_test = pd.DataFrame(data={'sents': sents, 'relations': relations, 'sent_ids': comments}) # added sentence ids to be used to group the same sentences for different pairs in joint ner ppi learning. 05-01-2021
	if dev is not None:
		sents, relations, comments, blanks = process_text(dev_text, 'dev')
		df_dev = pd.DataFrame(data={'sents': sents, 'relations': relations, 'sent_ids': comments}) # added sentence ids to be used to group the same sentences for different pairs in joint ner ppi learning. 05-01-2021
	
	# [GP][START] - use input parameter 'classes' instead of classes from the dataset because not every dataset contains all classes 
	#				in which case Relation_Mapper assigns overlapped relation ids. 04-02-2021
	#				E.g., BioCreative doesn't have a negative class -> {'enzyme': 0, 'structural': 1}
	#                     AImed, BioInfer -> {'enzyme': 0, 'structural': 1, 'negative': 2}
	#					  LLL doesn't have a structural class -> {'enzyme': 0, 'negative': 1} 
	#					  --> 'negative' id conflicts with the 'structural' id in BioCreative when the datasets are combined.
	#					  
	#rm = Relations_Mapper(pd.concat([df_train['relations'], df_dev['relations'], df_test['relations']], axis=0))
	rm = Relations_Mapper([x + '(e1,e2)' for x in classes])
	# [GP][END] - added a parameter to get a list of classes. 04-02-2021
	pickle.dump(rm, open(os.path.join(dir, 'relations_' + str(idx) + '.pkl'), "wb"))
	
	df_test['relations_id'] = df_test.progress_apply(lambda x: rm.rel2idx[x['relations']], axis=1)
	df_train['relations_id'] = df_train.progress_apply(lambda x: rm.rel2idx[x['relations']], axis=1)
	pickle.dump(df_train, open(os.path.join(dir, 'df_train_' + str(idx) + '.pkl'), "wb"))
	pickle.dump(df_test, open(os.path.join(dir, 'df_test_' + str(idx) + '.pkl'), "wb"))
	if dev is not None:
		df_dev['relations_id'] = df_dev.progress_apply(lambda x: rm.rel2idx[x['relations']], axis=1)
		pickle.dump(df_dev, open(os.path.join(dir, 'df_dev_' + str(idx) + '.pkl'), "wb"))
	else:
		df_dev = None

	return df_train, df_dev, df_test, rm
	

def preprocess_ppi(args):
	"""
	Data preprocessing for PPI datasets.
	
	History:
		- pre-processed BioCreative and BioCreative type datasets. 11-26-2020
		- pre-processed five PPI benchmark datasets: AImed, BioInfer, HPRD50, IEPA, LLL. 02-19-2021
	"""

	if args.do_cross_validation:
		# it used to retrieve a specific number of samples here, but since it reads a text from the beginning, it doesn't get random samples.
		# so, read the data all, and shuffle it and then get a specific number of samples. 12-23-2020
		# predefined_cls (if not set, it is None) is set when predefined lable is used instead of relation types from datasets. 01-06-2021
		if args.task == 'BioCreative':
			doc_samples = get_samples_from_biocreative(args.train_data, num_of_sample=None, predefined_relation_type=args.predefined_cls)
		elif args.task == 'BioCreative_type':
			doc_samples = get_samples_from_biocreative_type(args.train_data, num_of_sample=None, predefined_relation_type=args.predefined_cls)
		elif args.task == 'PPIbenchmark':
			doc_samples = get_samples_from_ppi_benchmark(args.train_data, num_of_sample=None, predefined_relation_type=args.predefined_cls)
		elif args.task == 'PPIbenchmark_type':
			doc_samples = get_samples_from_ppi_benchmark_type(args.train_data, num_of_sample=None, predefined_relation_type=args.predefined_cls)
		else:
			sys.exit('Unknown task!!')

		doc_samples = {k: v for k, v in doc_samples.items() if len(v) > 0} # remove documents having no samples.
		
		# debugging
		'''
		unique_samples = []
		for doc, samples in doc_samples.items():
			for s in samples:
				if s in unique_samples:
					print(s)
					input('enter..')
				else:
					unique_samples.append(s)
		'''

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
		'''
		print('len(unique_samples):', len(unique_samples))
		print('len(samples):', len(samples))
		
		samples_unique_samples = []
		for s in samples:
			for ss in s:
				if ss in samples_unique_samples:
					print(ss)
					input('enter..')
				else:
					samples_unique_samples.append(ss)
		print('len(samples_unique_samples):', len(samples_unique_samples))			

		all_test_samples = [] 
		'''

		# debugging
		if num_of_samples_for_eval != None and num_of_samples_for_eval != len([item for sublist in samples for item in sublist]):
			input('sampling number is wrong!!')
		
		if num_of_samples_for_eval == None:
			dir = args.train_data.rsplit('/', 1)[0] + '/all'
		else:
			dir = args.train_data.rsplit('/', 1)[0] + '/' + str(args.num_samples)
		
		if not os.path.exists(dir):
			os.makedirs(dir)

		samples = np.array(samples)
		kfold = KFold(n_splits=args.num_of_folds, shuffle=False)
		for idx, (train_index, test_index) in enumerate(kfold.split(samples)):
			
			## Train/Validation(Dev)-optional/Test split - k-folds (train/test split), 80/10/10, 70/15/15, 60/20/20 ratio
			if args.ratio == 'k-folds':
				pass
			elif args.ratio == '80-10-10':
				dev_index = train_index[:(len(train_index)//9)]
				train_index = train_index[len(dev_index):]
			
				#TODO: fix this! - there are duplicates among test sets that must not have duplicates. 04-21-2021
				''' 
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
				'''
			else:
				logger.error("Unknown ratio!!")
				sys.exit()
				
			if args.ratio == 'k-folds':
				train, test = samples[train_index], samples[test_index]
				dev = None
				#print("TRAIN len:", len(train_index), "TEST len:", len(test_index))
				#print("TRAIN:", train_index, "TEST:", test_index)
			else:
				train, dev, test = samples[train_index], samples[dev_index], samples[test_index]
				#print("TRAIN len:", len(train_index), "DEV len:", len(dev_index), "TEST len:", len(test_index))
				#print("TRAIN:", train_index, "DEV:", dev_index, "TEST:", test_index)
			
			# debug
			'''
			#test_samples = set()
			for x in test:
				for y in x:
					if y[0] in all_test_samples:
						print('duplicate:', y[0])
						input('enter..')
						
					else:
						test_samples.append(y[0])
						
			train_samples = set()
			for x in train:
				for y in x:
					train_samples.add(y[0])

			#dev_samples = set()
			#for x in dev:
			#	for y in x:
			#		dev_samples.add(y[0])

			#for i in train_samples.intersection(dev_samples):
			#	print('train_dev overlap:', i)
			
			#for i in train_samples.intersection(test_samples):
			#	print('train_test overlap:', i)
			
			#for i in dev_samples.intersection(test_samples):
			#	print('dev_test overlap:', i)
			
			print('len(train_samples):', len(train_samples))
			#print('len(dev_samples):', len(dev_samples))
			print('len(test_samples):', len(test_samples))
			
			#input('enter.....')
			'''

			df_train, df_dev, df_test, rm = store_data(dir, train, dev, test, args.classes, idx)	
			
			if idx == 0:
				first_df_train = df_train
				first_df_dev = df_dev
				first_df_test = df_test
				first_rm = rm
			
		logger.info("Finished and saved!")
		
		input('enter..')
		
		return first_df_train, first_df_dev, first_df_test, first_rm # return the first CV set.

	else:
		if args.task == 'BioCreative':
			train_samples = get_samples_from_biocreative(args.train_data, num_of_sample=None, predefined_relation_type=args.predefined_cls)
			if args.train_data != args.test_data:
				test_samples = get_samples_from_biocreative(args.test_data, num_of_sample=None, predefined_relation_type=args.predefined_cls)
		elif args.task == 'BioCreative_type':
			train_samples = get_samples_from_biocreative_type(args.train_data, num_of_sample=None, predefined_relation_type=args.predefined_cls)
			if args.train_data != args.test_data:
				test_samples = get_samples_from_biocreative_type(args.test_data, num_of_sample=None, predefined_relation_type=args.predefined_cls)
		elif args.task == 'PPIbenchmark':
			train_samples = get_samples_from_ppi_benchmark(args.train_data, num_of_sample=None, predefined_relation_type=args.predefined_cls)
			if args.train_data != args.test_data:
				test_samples = get_samples_from_ppi_benchmark(args.test_data, num_of_sample=None, predefined_relation_type=args.predefined_cls)
		elif args.task == 'PPIbenchmark_type':
			train_samples = get_samples_from_ppi_benchmark_type(args.train_data, num_of_sample=None, predefined_relation_type=args.predefined_cls)
			if args.train_data != args.test_data:
				test_samples = get_samples_from_ppi_benchmark_type(args.test_data, num_of_sample=None, predefined_relation_type=args.predefined_cls)
		else:
			sys.exit('Unknown task!!')
			
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
		
		if not os.path.exists(dir):
			os.makedirs(dir)
			
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
		
		
		# [GP]
		'''
		print(self.df['input'].iloc[0])
		print(tokenizer.decode(self.df['input'].iloc[0]))
		print(tokenizer.convert_ids_to_tokens(self.df['input'].iloc[0]))
		print(self.df['e1_e2_start'].iloc[0])
		input('enter..')
		'''
		
	
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
			#			  - pre-processed BioCreative, BioCreative_type data. 11-26-2020
			# 			  - added dev set. 12-23-2020
			#			  - pre-processed PPIbenchmark, PPIbenchmark_type incl. AImed, BioInfer, HPRD50, IEPA, LLL. 02-19-2021
			if os.path.isfile(args.train_data) and args.train_data.endswith('.pkl') == False: # preprocess original txt, json, tsv files.
				df_train, df_dev, df_test, rm = preprocess_ppi(args)
			else:
				relations_path = os.path.join(args.train_data, 'relations_' + str(dataset_num) + '.pkl')
				train_path = os.path.join(args.train_data, 'df_train_' + str(dataset_num) + '.pkl')
				test_path = os.path.join(args.train_data, 'df_test_' + str(dataset_num) + '.pkl')
				dev_path = os.path.join(args.train_data, 'df_dev_' + str(dataset_num) + '.pkl')

				if os.path.isfile(relations_path) and os.path.isfile(train_path) and os.path.isfile(test_path):
					rm = pickle.load(open(relations_path, "rb"))
					df_train = pickle.load(open(train_path, "rb"))
					df_test = pickle.load(open(test_path, "rb"))
					df_dev = pickle.load(open(dev_path, "rb")) if os.path.isfile(dev_path) else None # optional

					logger.info("Loaded preproccessed data.")
				else:
					logger.error("Data files do not exist!!")
					sys.exit()
			# [GP][END] - PPI datasets pre-processing.

		train_set = semeval_dataset(df_train, tokenizer=tokenizer, e1_id=e1_id, e2_id=e2_id)
		test_set = semeval_dataset(df_test, tokenizer=tokenizer, e1_id=e1_id, e2_id=e2_id)
		train_length = len(train_set); test_length = len(test_set)
		# [GP][START] - added dev set. 12-23-2020
		if df_dev is not None:
			dev_set = semeval_dataset(df_dev, tokenizer=tokenizer, e1_id=e1_id, e2_id=e2_id)
			dev_length = len(dev_set)
		else:
			dev_length = 0
		# [GP][END] - added dev set. 12-23-2020
		PS = Pad_Sequence(seq_pad_value=tokenizer.pad_token_id,\
						  label_pad_value=tokenizer.pad_token_id,\
						  label2_pad_value=-1)
		train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, \
								  num_workers=0, collate_fn=PS, pin_memory=False)
		test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, \
								  num_workers=0, collate_fn=PS, pin_memory=False)
		# [GP][START] - added dev set. 12-23-2020
		if df_dev is not None:
			dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=True, \
									num_workers=0, collate_fn=PS, pin_memory=False)
		else:
			dev_loader = None
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
