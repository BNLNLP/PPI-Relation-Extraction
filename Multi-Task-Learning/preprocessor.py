"""
The code is modified based on from preprocessing_func.py (PPI-Relation-Extraction/BERT-Relation-Extraction). 11-04-2021

"""
import os
import sys
import re
import random
import numpy as np
import json
import csv
import logging

from lxml import etree
from argparse import ArgumentParser
from sklearn.model_selection import KFold, train_test_split

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')


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


def get_samples_from_biocreative_type(file, num_of_sample=None):
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
			
		if relation_type not in ['enzyme', 'structural']: # exclude 'misc' for now since they are very few (only 3 as of 12-17-2020)
			logger.info('this relation_type is undefined: %s', sentence)
			continue
			
		relation = relation.split(';')
		rel_pairs = []
		for rel in relation:
			entities = re.split(' -> | - | \? ', rel)
			entities = [x.strip() for x in entities]
			entities = [x.replace('_', ' ') for x in entities]
			
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
					tagged_sent = sentence.replace(e1, '[E1]' + e1 + '[/E1]', 1)
					if sentence.startswith('Yeast epiarginase regulation,'):
						tagged_sent = tagged_sent.replace('and arginase responsible', 'and [E2]arginase[/E2] responsible', 1)
					else:
						tagged_sent = tagged_sent.replace(e2, '[E2]' + e2 + '[/E2]', 1)
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
					e1_s_idx = tagged_sent.index('[E1]')
					e1_e_idx = tagged_sent.index('[/E1]')
					e2_s_idx = tagged_sent.index('[E2]')
					e2_e_idx = tagged_sent.index('[/E2]')
					
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
					
					relation_id = 0 if relation_type == 'enzyme' else 1
					
					sample = {'pair_id': pair_id,
							  'sent_id': sent_id, # pair_id and sent_id are the same in BioCreative Typed annotation.
							  'entity_marked_sent': tagged_sent,
							  'relation': relation_type,
							  'relation_id': relation_id,
							  'directed': False, # relation directionality. a.k.a symmetric or asymmetric relation.
							  'reverse': False} # this is only used for undirected relations. 
												# For testing phase, undirected samples are replicated, and the replicated samples are tagged as reverse. 
												# So, if it's set to true, the model uses the second entity + the first entity instead of 
												# the first entity + the second entity to classify both relation representation cases (A + B, B + A). 

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


def get_samples_from_biocreative(file, num_of_sample=None):
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
							tagged_text = tagged_text[:g2_offset] + '[E2]' + g2_text + '[/E2]' + tagged_text[g2_offset + g2_length:]
							tagged_text = tagged_text[:g1_offset] + '[E1]' + g1_text + '[/E1]' + tagged_text[g1_offset + g1_length:]
						else:
							tagged_text = tagged_text[:g1_offset] + '[E1]' + g1_text + '[/E1]' + tagged_text[g1_offset + g1_length:]
							tagged_text = tagged_text[:g2_offset] + '[E2]' + g2_text + '[/E2]' + tagged_text[g2_offset + g2_length:]
						
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
def get_samples_from_ppi_benchmark(file):
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
					tagged_sent = tagged_sent[:e2_start_idx] + '[E2]' + e2_text + '[/E2]' + tagged_sent[e2_end_idx:]
					tagged_sent = tagged_sent[:e1_start_idx] + '[E1]' + e1_text + '[/E1]' + tagged_sent[e1_end_idx:]
				else:
					tagged_sent = tagged_sent[:e1_start_idx] + '[E1]' + e1_text + '[/E1]' + tagged_sent[e1_end_idx:]
					tagged_sent = tagged_sent[:e2_start_idx] + '[E2]' + e2_text + '[/E2]' + tagged_sent[e2_end_idx:]
				
				if tagged_sent in unique_sents: # all sentences with tags must be unique. this is for just in case.
					print('duplicate sent:', tagged_sent)
					continue
				else:
					unique_sents.add(tagged_sent)

				relation_type = 'positive' if pair_interaction == 'True' else 'negative'
				relation_id = 0 if pair_interaction == 'True' else 1

				sample = {'pair_id': pair_id,
						  'sent_id': sent_id,
						  'entity_marked_sent': tagged_sent,
						  'relation': relation_type,
						  'relation_id': relation_id,
						  'directed': False, # relation directionality. a.k.a symmetric or asymmetric relation.
						  'reverse': False} # this is only used for undirected relations. 
											# For testing phase, undirected samples are replicated, and the replicated samples are tagged as reverse. 
											# So, if it's set to true, the model uses the second entity + the first entity instead of 
											# the first entity + the second entity to classify both relation representation cases (A + B, B + A). 

				if doc_id in samples:
					samples[doc_id].append(sample)
				else:
					samples[doc_id] = [sample]

	return samples
# [GP][END] - PPI benchmark datasets pre-processing. 02-19-2021


def get_samples_from_ppi_benchmark_type(file, neg_sample_processing=False):
	"""
	Data preprocessing for PPI benchmark datasets typed (annotated by BNL). 03-18-2021
	
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
				sample_pair_id = sample['pair_id']
				if pair_id == sample_pair_id:
					samples[doc_id][idx]['relation'] = relation_type
					samples[doc_id][idx]['relation_id'] = 0 if relation_type == 'enzyme' else 1
					return
		return
	
	# [START] used for negative_annotation_files e.g., passed_full_aimed_training.tsv 05-04-2021	
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
				sample_pair_id = sample['pair_id']
				if sample_pair_id in negative_pair_ids:
					sample['relation_id'] = 2
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
			samples[doc_id] = [x for x in sample_list if not x['relation'] == 'positive']
		# [END] used for negative_annotation_files e.g., passed_full_aimed_training.tsv 05-04-2021
		else:
			# [START] use this option to remove 'negative' samples from original data. 05-04-2021
			samples[doc_id] = [x for x in sample_list if not x['relation'] == 'positive' and not x['relation'] == 'negative']
			# [END] use this option to remove 'negative' samples from original data. 05-04-2021
			
	return samples


def store_data(dir, train, dev, test, idx=0):
	"""
	ref: https://pythonhowtoprogram.com/how-to-write-multiple-json-objects-to-a-file-in-python-3/
	"""
	train_txt = ''
	for doc_samples in train:
		for sample in doc_samples:
			train_txt += json.dumps(sample)
			train_txt += '\n'
	
	outfile = os.path.join(dir, 'train_' + str(idx) + '.json')
	with open(outfile, "w") as f: 
		f.write(train_txt)
	
	
	"""
	For development (validation) and test data, undirectional relation samples are replicated with reverse = True,
	so that the model classifies both relation representation cases (A + B, B + A). 
	"""
	if dev is not None:
		dev_txt = ''
		for doc_samples in dev:
			for sample in doc_samples:
				dev_txt += json.dumps(sample)
				dev_txt += '\n'
				
				if sample['directed'] is False:
					sample['reverse'] = True
					dev_txt += json.dumps(sample)
					dev_txt += '\n'
		
		outfile = os.path.join(dir, 'dev_' + str(idx) + '.json')
		with open(outfile, "w") as f: 
			f.write(dev_txt)
	
	test_txt = ''
	for doc_samples in test:
		for sample in doc_samples:
			test_txt += json.dumps(sample)
			test_txt += '\n'
			
			if sample['directed'] is False:
				sample['reverse'] = True
				test_txt += json.dumps(sample)
				test_txt += '\n'

	outfile = os.path.join(dir, 'test_' + str(idx) + '.json')
	with open(outfile, "w") as f: 
		f.write(test_txt)


def preprocess_ppi(data, args, neg_sample_processing):
	"""
	Data preprocessing for PPI datasets.
	
	History:
		- pre-processed BioCreative and BioCreative type datasets. 11-26-2020
		- pre-processed five PPI benchmark datasets: AImed, BioInfer, HPRD50, IEPA, LLL. 02-19-2021
	"""

	if args.do_cross_validation:
		# it used to retrieve a specific number of samples here, but since it reads a text from the beginning, it doesn't get random samples.
		# so, read the data all, and shuffle it and then get a specific number of samples. 12-23-2020
		if args.task == 'BioCreative':
			doc_samples = get_samples_from_biocreative(data)
		elif args.task == 'BioCreative_type':
			doc_samples = get_samples_from_biocreative_type(data)
		elif args.task == 'PPIbenchmark':
			doc_samples = get_samples_from_ppi_benchmark(data)
		elif args.task == 'PPIbenchmark_type':
			doc_samples = get_samples_from_ppi_benchmark_type(data, neg_sample_processing)
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
			dir = data.rsplit('/', 1)[0] + '/all'
		else:
			dir = data.rsplit('/', 1)[0] + '/' + str(args.num_samples)
		
		if neg_sample_processing:
			dir += '_negative'
				
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
			
			store_data(dir, train, dev, test, idx)	
		
		logger.info("Finished and saved!")


def main():
	parser = ArgumentParser()
	
	# [GP][START] - added PPI tasks.
	#			  - added BioCreative, BioCreative_type. 11-26-2020
	#			  - added PPIbenchmark, PPIbenchmark_type incl. AImed, BioInfer, HPRD50, IEPA, LLL. 02-19-2021
	parser.add_argument("--task", type=str, default='PPIbenchmark', help='PPIbenchmark, PPIbenchmark_type, \
																		  BioCreative, BioCreative_type')
	parser.add_argument("--data_list", nargs="+", default=['./datasets/ppi/type_annotation/AImed_type/aimed_type_annotations_srm.tsv, \
														    ./datasets/ppi/type_annotation/AImed_type/passed_full_aimed_training_ikb.tsv'], \
														    help="data file path list")
	
	# [GP][START] - arguments for cross-validation. 11-29-2020
	parser.add_argument("--num_samples", type=int, default=-1, help="Number of samples for the model. -1 means to use all samples.")
	parser.add_argument("--do_cross_validation", action="store_true", help="Whether to use cross-validation for evaluation.")
	parser.add_argument("--num_of_folds", default=10, type=int, help="The number of folds for the cross validation.")
	parser.add_argument("--ratio", type=str, default='k-folds', help="k-folds generates train-test splits given the number of given folds. \
																	  if you want to use validation (development) set, use one of the following ratios: \
																	  train/dev/test ratio: 80-10-10, 70-15-15, 60-20-20")
	# [GP][END] - arguments for cross-validation. 11-29-2020

	args = parser.parse_args()
	
	for idx, data in enumerate(args.data_list):
		# this parameter is only used for typed ppi benchmark datasets (1st data: positive(enzyme, structurel), 2nd data: negative)
		neg_sample_processing = True if idx == 1 else False 
		preprocess_ppi(data, args, neg_sample_processing)
	
	
if __name__ == "__main__":
    main()	
