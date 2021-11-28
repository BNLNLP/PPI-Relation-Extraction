"""
This code converts the old data format (pickle for DataFrame) to the JSON format. 11-04-2021

"""
import os
import pickle
import pandas as pd
import json
from lxml import etree
import spacy
from spacy.tokens import Doc



#pkl_dir = 'datasets/ppi/original/AImed/all_ver_3'
#pkl_dir = 'datasets/ppi/original/BioInfer/all_ver_3'
pkl_dir = 'datasets/ppi/original/HPRD50/all'
#pkl_dir = 'datasets/ppi/original/IEPA/all_ver_1'
#pkl_dir = 'datasets/ppi/original/LLL/all_ver_1'
#pkl_dir = 'datasets/ppi/type_annotation/ALL/all_incl_negative_annotation_ver_17'


# read entity types.
#xml_file = 'datasets/ppi/original/AImed/AImed.xml'
#xml_file = 'datasets/ppi/original/BioInfer/BioInferCLAnalysis_split_SMBM_version.xml'
xml_file = 'datasets/ppi/original/HPRD50/HPRD50.xml'
#xml_file = 'datasets/ppi/original/IEPA/IEPA.xml'
#xml_file = 'datasets/ppi/original/LLL/LLL.xml'

xml_parser = etree.XMLParser(ns_clean=True)
tree = etree.parse(xml_file, xml_parser)
root = tree.getroot()

entity_type_dict = {}
unique_entity_types = {} # to create entity type file.

for doc_elem in root.findall('.//document'):
	doc_id = doc_elem.get('id')
	pubmed_id = doc_elem.get('origId')

	for sent_elem in doc_elem.findall('.//sentence'):
		sent_id = sent_elem.get('id')
		sent_txt = sent_elem.get('text')
		
		entity_type_dict[sent_id] = {}
		
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
			
			if ent_char_offset in entity_type_dict[sent_id]:
				print('ERROR - duplicate charOffset!!')
				print(sent_id)
				input('enter..')
			else:
				if ent_type is None: # for instance, IEPA doen't have types.
					ent_type = 'no-type'
					
				entity_type_dict[sent_id][ent_char_offset] = (ent_text, ent_type)
				if ent_type not in unique_entity_types:
					unique_entity_types[ent_type] = {'id': len(unique_entity_types)}
					
			#if sent_id == "HPRD50.d33.s1.e1" or sent_id == "HPRD50.d42.s5":
			#	print(sent_id)
			#	print(entity_type_dict[sent_id])
			#	input('enter..oweieopwi')
				
				
'''
for k, v in entity_type_dict.items():
	print(k, v)

et_dict = {}
for t in unique_entity_types:
	print(t)
'''

entity_file = xml_file.rsplit('/', 1)[0] + '/entity_types.json'
if not os.path.exists(entity_file):
	with open(entity_file, 'w') as fp:
		json.dump(unique_entity_types, fp)




nlp = spacy.load('en_core_web_sm')


# debug
total_num_of_samples = 0
num_of_samples_with_overlapped_entities = 0
num_of_samples_with_no_context_between_entities = 0
num_of_samples_with_no_predicate_in_context = 0 # count the number of sentences where no verb exists between the two entities.
num_of_samples_with_entity_in_predicate = 0 # count the number of sentences where predicate contains entity because of parsing error.
num_of_predicates_for_samples_with_no_context_between_entities = 0 # count the samples where a predicate having either entity as child exists for samples that no verb exists between the two entities.
num_of_samples_with_error_tags = 0 # count the number of samples where tag error exists such as more than two ' >>'.



for pkl_file in os.listdir(pkl_dir):
	if not pkl_file.startswith('df_'): # skip relations files.
		continue
		
	df = pd.read_pickle(os.path.join(pkl_dir, pkl_file))

	output_txt = ''
	for idx, row in df.iterrows():		
		text = row['sents']
		rel_type = row['relations'].replace('(e1,e2)', '')
		rel_id = row['relations_id']
		sent_id = row['sent_ids']
		
		'''
		print(row)
		print(entity_marked_sent)
		print(relation)
		print(relation_id)
		print(sent_id)
		input('enter...')
		'''
		
		
		
		
		
		
		
		
		# debug
		e1_s_t = text.count('[E1]')
		e1_e_t = text.count('[/E1]')
		e2_s_t = text.count('[E2]')
		e2_e_t = text.count('[/E2]')
		if e1_s_t != 1 or e1_e_t != 1 or e2_s_t != 1 or e2_e_t != 1:
			print(e1_s_t, e1_e_t, e2_s_t, e2_e_t)
			print(text)
			num_of_samples_with_error_tags += 1
			continue

		e1_start = text.index('[E1]') + 4
		e1_end   = text.index('[/E1]')
		e2_start = text.index('[E2]') + 4
		e2_end   = text.index('[/E2]')

		entity_1 = text[e1_start:e1_end]
		entity_2 = text[e2_start:e2_end]
		
		
		

		
		# these are needed to find the original offset of entities. no additional space is not considered. this is used for finding entity types by charOffset. 
		orig_e1_start = e1_start
		orig_e1_end = e1_end
		orig_e2_start = e2_start
		orig_e2_end = e2_end
	
		
				
		adjusted_offset_for_e1_start = 4 # 4 is the length of '[E1]'
		adjusted_offset_for_e1_end   = 5 # 5 is the length of '[/E1]'
		adjusted_offset_for_e2_start = 4 # 4 is the length of '[E2]'
		adjusted_offset_for_e2_end   = 5 # 3 is the length of '[/E2]'
		
		if text.index('[E1]') != 0 and text[text.index('[E1]')-1] != ' ':
			text = text.replace('[E1]', ' ')
			adjusted_offset_for_e1_start -= 1
		else:
			text = text.replace('[E1]', '')

		if text.index('[/E1]') != len(text)-5 and text[text.index('[/E1]')+5] != ' ':
			text = text.replace('[/E1]', ' ')
			adjusted_offset_for_e1_end -= 1
		else:
			text = text.replace('[/E1]', '')

		if text.index('[E2]') != 0 and text[text.index('[E2]')-1] != ' ':
			text = text.replace('[E2]', ' ')
			adjusted_offset_for_e2_start -= 1
		else:
			text = text.replace('[E2]', '')
		
		if text.index('[/E2]') != len(text)-5 and text[text.index('[/E2]')+5] != ' ':
			text = text.replace('[/E2]', ' ')
			adjusted_offset_for_e2_end -= 1
		else:
			text = text.replace('[/E2]', '')

		final_adjusted_offset_for_e1_start = 0 	
		final_adjusted_offset_for_e1_end = 0 	
		final_adjusted_offset_for_e2_start = 0 	
		final_adjusted_offset_for_e2_end = 0 	
		
		# these are needed to find the original offset of entities. no additional space is not considered. this is used for finding entity types by charOffset. 
		final_adjusted_offset_for_orig_e1_start = 0 	
		final_adjusted_offset_for_orig_e1_end = 0 	
		final_adjusted_offset_for_orig_e2_start = 0 	
		final_adjusted_offset_for_orig_e2_end = 0 	

		l = [e1_start, e1_end, e2_start, e2_end]
		for num, i in enumerate(sorted(l)):
			if i == e1_start:
				if num == 0:
					final_adjusted_offset_for_e1_start = adjusted_offset_for_e1_start
					final_adjusted_offset_for_orig_e1_start = 4
				elif num == 1:
					final_adjusted_offset_for_e1_start = adjusted_offset_for_e1_start + adjusted_offset_for_e2_start
					final_adjusted_offset_for_orig_e1_start = 4 + 4
				elif num == 2:
					final_adjusted_offset_for_e1_start = adjusted_offset_for_e1_start + adjusted_offset_for_e2_start + adjusted_offset_for_e2_end
					final_adjusted_offset_for_orig_e1_start = 4 + 4 + 5
			elif i == e1_end:
				if num == 1:
					final_adjusted_offset_for_e1_end = adjusted_offset_for_e1_start
					final_adjusted_offset_for_orig_e1_end = 4
				elif num == 2:
					final_adjusted_offset_for_e1_end = adjusted_offset_for_e1_start + adjusted_offset_for_e2_start
					final_adjusted_offset_for_orig_e1_end = 4 + 4
				elif num == 3:
					final_adjusted_offset_for_e1_end = adjusted_offset_for_e1_start + adjusted_offset_for_e2_start + adjusted_offset_for_e2_end
					final_adjusted_offset_for_orig_e1_end = 4 + 4 + 5
			elif i == e2_start:
				if num == 0:
					final_adjusted_offset_for_e2_start = adjusted_offset_for_e2_start
					final_adjusted_offset_for_orig_e2_start = 4
				elif num == 1:
					final_adjusted_offset_for_e2_start = adjusted_offset_for_e2_start + adjusted_offset_for_e1_start
					final_adjusted_offset_for_orig_e2_start = 4 + 4
				elif num == 2:
					final_adjusted_offset_for_e2_start = adjusted_offset_for_e2_start + adjusted_offset_for_e1_start + adjusted_offset_for_e1_end
					final_adjusted_offset_for_orig_e2_start = 4 + 4 + 5
			elif i == e2_end:
				if num == 1:
					final_adjusted_offset_for_e2_end = adjusted_offset_for_e2_start
					final_adjusted_offset_for_orig_e2_end = 4
				elif num == 2:
					final_adjusted_offset_for_e2_end = adjusted_offset_for_e2_start + adjusted_offset_for_e1_start
					final_adjusted_offset_for_orig_e2_end = 4 + 4
				elif num == 3:
					final_adjusted_offset_for_e2_end = adjusted_offset_for_e2_start + adjusted_offset_for_e1_start + adjusted_offset_for_e1_end
					final_adjusted_offset_for_orig_e2_end = 4 + 4 + 5
					
		e1_start -= final_adjusted_offset_for_e1_start
		e1_end   -= final_adjusted_offset_for_e1_end
		e2_start -= final_adjusted_offset_for_e2_start
		e2_end   -= final_adjusted_offset_for_e2_end
		

		# debug
		n1 = text[e1_start:e1_end] # debug
		n2 = text[e2_start:e2_end] # debug
		if entity_1 != n1 or entity_2 != n2: # they must be the same.
			print(text)
			print('e1:', entity_1, '/ e2:', entity_2)
			print('n1:', n1, '/ n2:', n2)
			input('enter...')

		
		
		
		
		
		# these are needed to find the original offset of entities. no additional space is not considered. this is used for finding entity types by charOffset. 
		orig_e1_start -= final_adjusted_offset_for_orig_e1_start
		orig_e1_end   -= final_adjusted_offset_for_orig_e1_end
		orig_e2_start -= final_adjusted_offset_for_orig_e2_start
		orig_e2_end   -= final_adjusted_offset_for_orig_e2_end

		# find entity types.
		e1_offset = str(orig_e1_start) + '-' + str(orig_e1_end-1)
		e2_offset = str(orig_e2_start) + '-' + str(orig_e2_end-1)
			
		entity_1_type = entity_type_dict[sent_id][e1_offset][1]
		entity_2_type = entity_type_dict[sent_id][e2_offset][1]
		
		# debug
		if entity_1 != entity_type_dict[sent_id][e1_offset][0] or entity_2 != entity_type_dict[sent_id][e2_offset][0]:
			print('Error - entity names do not match!!')
			print('sent_id:', sent_id)
			print('row[sents]:', row['sents'])
			print('entity_type_dict[sent_id]:', entity_type_dict[sent_id])
			print('e1_offset:', e1_offset)
			print('e2_offset:', e2_offset)
			input('enter..')
		

		
		
		
		doc = nlp(text)
		
		#for token in doc:
		#	children = [child.text for child in token.children]
		#	print(token.text, token.dep_, token.head.text, token.head.pos_, children)
		
		
		tokens = [x.text for x in doc]
		
		# debug
		'''
		if text.startswith('Dasatinib') or text.startswith('Mitiglinide'):
			print('>> original text:', line['text'])
			print('>> marker free text:', text)
			print('>> tokens:', tokens)
			print('>> entities:', entity_1, '(', e1_start, ',', e1_end, ')', entity_2, '(', e2_start, ',', e2_end, ')')
			print('>> rel_type:', rel_type)
			input('enter..')
		'''	

		predicates_elems = {}
		predicates_indices = []
		
		#verb_elems = []
		#verb_indices = []
		
		
		root_verb = {}
		verbs = []
		
		#print('======================================================================================')
		
		# debug
		is_e1_start_updated = False
		is_e1_end_updated = False
		is_e2_start_updated = False
		is_e2_end_updated = False
		
		for idx, token in enumerate(doc):
			
			#print(token, token.i)
			
			token_text = token.text
			t_s = token.idx # token start index
			t_e = token.idx + len(token) # token end index

			if len(token_text.strip()) == 0: # skip empty tokens.
				continue

			# debug
			assert token.text == text[t_s:t_e], 'two tokens must be the same!!'
			
			if t_s == e1_start:
				e1_start = idx
				is_e1_start_updated = True
			
			if t_s == e2_start:
				e2_start = idx
				is_e2_start_updated = True
			
			if t_e == e1_end:
				e1_end = idx + 1
				is_e1_end_updated = True
			
			if t_e == e2_end:
				e2_end = idx + 1
				is_e1_end_updated = True
			
			
			
			
			if token.dep_ == 'ROOT' and token.pos_ == 'VERB':
				#predicates_elems[token.i] = token
				#root_index = token.i
				#predicates_indices.append(token.i)

				#print('>> ROOT verb:', token, '/ idx:', token.i)
				root_verb = {'verb_idx': [token.i], 'verb_token': {token.i: token}}
			
			
			if token.pos_ == 'VERB':
				#verb_indices.append(token.i)
				#print('>> verb:', token, '/ idx:', token.i)
				verbs.append({'verb_idx': [token.i], 'verb_token': {token.i: token}})
				
		# debug
		if all([is_e1_start_updated, is_e1_end_updated, is_e2_start_updated, is_e2_end_updated]):
			print(tokens)
			print(e1_start, e1_end, e2_start, e2_end)
			print('Error!! - Not all entity indexes are updated.')
			input('enter...')
	
		
		# debug - counting overlapped entities
		if (e1_start >= e2_start and e1_start <= e2_end) or (e2_start >= e1_start and e2_start <= e1_end):
			num_of_samples_with_overlapped_entities += 1	
		
		# debug - counting no context (tokens between entities)
		if e2_start == e1_end or e1_start == e2_end:
			num_of_samples_with_no_context_between_entities += 1
		
		# debug - error checking
		if e1_end <= e1_start or e2_end <= e2_start:
			print(text)
			print(entity_1)
			print(tokens[e1_start:e1_end])
			print(entity_2)
			print(tokens[e2_start:e2_end])
			input('enter..')
		
		
		
		if len(root_verb) != 0:
			root_verb_index = root_verb['verb_idx'][0]
			for child in doc[root_verb_index].children:
				if child.dep_ == 'aux' or child.dep_ == 'auxpass' or child.dep_ == 'neg':
					root_verb['verb_idx'].append(child.i)
					root_verb['verb_token'][child.i] = child.text
			
		
		for v in verbs:
			v_idx = v['verb_idx'][0]
			children_indice = []
			children_tokens = []
			for child in doc[v_idx].children:
				children_indice.append(child.i)
				children_tokens.append(child.text)
				
				if child.dep_ == 'aux' or child.dep_ == 'auxpass' or child.dep_ == 'neg':
					v['verb_idx'].append(child.i)
					v['verb_token'][child.i] = child.text
					
			v['children_indice'] = children_indice
			v['children_tokens'] = children_tokens
		
		
		#for v in verbs:
		#	print(v)
			

		
		'''
		items = list(predicates_elems.keys())
		items.sort()
		predicates = ' '.join([str(predicates_elems[ind]) for ind in items])
		print(">> the root predicates is:", predicates)
		
		predicate_start_idx = min(predicates_indices)
		predicate_end_idx = max(predicates_indices) + 1
		#input('enter..')
		'''

		predicate_exists_between_entities = False
		for v in verbs:
			v_idx = v['verb_idx'][0] # main verb index
			
			is_e1_child = False
			is_e2_child = False
			
			for c_i in v['children_indice']:
				if (c_i >= e1_start and c_i < e1_end):
					is_e1_child = True
				if (c_i >= e2_start and c_i < e2_end):
					is_e2_child = True
			
			if is_e1_child and is_e2_child:
				if e1_start < e2_start: # e1 appears before e2
					if v_idx >= e1_end and v_idx < e2_start:
						predicate_exists_between_entities = True
						break
				elif e1_start > e2_start: # e2 appears before e1
					if v_idx >= e2_end and v_idx < e1_start:
						predicate_exists_between_entities = True
						break

		if predicate_exists_between_entities == False:
			num_of_samples_with_no_predicate_in_context += 1
		
		# debug - counting samples where entitiy exists in predicate.
		'''
		if (predicate_start_idx <= e1_start and e1_start < predicate_end_idx) or \
		   (predicate_start_idx <= e1_end-1 and e1_end-1 < predicate_end_idx) or \
		   (predicate_start_idx <= e2_start and e2_start < predicate_end_idx) or \
		   (predicate_start_idx <= e2_end-1 and e2_end-1 < predicate_end_idx):
		   num_of_samples_with_entity_in_predicate += 1
		'''

		
		predicates = []
		predicates_idx = []
		

		#print(">> the root predicates is:", predicates, '/ index: (', predicate_start_idx, ',', predicate_end_idx, ') / predicate using index:', tokens[predicate_start_idx:predicate_end_idx])
		
		#if predicates != ' '.join(tokens[predicate_start_idx:predicate_end_idx]):
		#	input('enter..')
		
		#print('>> tokens:', tokens)
		#print('>> entities:', entity_1, '(', e1_start, ',', e1_end, ')', entity_2, '(', e2_start, ',', e2_end, ')')
		#print('>> rel_type:', rel_type)
		
		
		use_predicate_span = False
			
		if predicate_exists_between_entities == False:
		#if True:
			num_of_found_predicates = 0 # debug
			
			for v in verbs:
				items = list(v['verb_token'].keys())
				items.sort()
				predicate = ' '.join([str(v['verb_token'][ind]) for ind in items])
									
				#print('verb_token:', predicates)
				
				'''
				for c_i in v['children_indice']:
					if (c_i >= e1_start and c_i < e1_end) or (c_i >= e2_start and c_i < e2_end):
						#print('c_i, doc[c_i].text:', c_i, doc[c_i].text)
						#input('enter..')
						
						predicate_start_idx = min(v['verb_idx'])
						predicate_end_idx = max(v['verb_idx']) + 1
						
						if not ((predicate_start_idx >= e1_end and predicate_start_idx < e2_start) and \
								(predicate_end_idx >= e1_end and predicate_end_idx < e2_start)):
							use_predicate_span = True
							num_of_predicates_for_samples_with_no_context_between_entities += 1
							
							num_of_found_predicates += 1
							
							predicates.append(predicate)
							predicates_idx.append((predicate_start_idx, predicate_end_idx))
							
							break
				'''
				
				is_e1_child = False
				is_e2_child = False
				
				for c_i in v['children_indice']:
					if (c_i >= e1_start and c_i < e1_end):
						is_e1_child = True
					if (c_i >= e2_start and c_i < e2_end):
						is_e2_child = True
						
				if is_e1_child or is_e2_child:
					predicate_start_idx = min(v['verb_idx'])
					predicate_end_idx = max(v['verb_idx']) + 1
					
					if e1_start < e2_start: # e1 appears before e2 
						''' [predicate] [e1] ... [e2] or [e1] ... [e2] [predicate] '''
						if predicate_end_idx <= e1_start or predicate_start_idx >= e2_end:
							#if not ((predicate_start_idx >= e1_end and predicate_start_idx < e2_start) and \
							#		(predicate_end_idx > e1_end and predicate_end_idx <= e2_start)):
							use_predicate_span = True
							num_of_predicates_for_samples_with_no_context_between_entities += 1
							
							num_of_found_predicates += 1
							
							predicates.append(predicate)
							predicates_idx.append((predicate_start_idx, predicate_end_idx))
					elif e1_start > e2_start: # e2 appears before e1
						''' [predicate] [e2] ... [e1] or [e2] ... [e1] [predicate] '''
						if predicate_end_idx <= e2_start or predicate_start_idx >= e1_end:
							#if not ((predicate_start_idx >= e2_end and predicate_start_idx < e1_start) and \
							#		(predicate_end_idx > e2_end and predicate_end_idx <= e1_start)):
							use_predicate_span = True
							num_of_predicates_for_samples_with_no_context_between_entities += 1
							
							num_of_found_predicates += 1
							
							predicates.append(predicate)
							predicates_idx.append((predicate_start_idx, predicate_end_idx))

			if num_of_found_predicates > 1:
				print(num_of_found_predicates)

		
		# add ROOT predicate
		'''
		if len(root_verb) != 0:
			items = list(root_verb['verb_token'].keys())
			items.sort()
			root_predicate = ' '.join([str(root_verb['verb_token'][ind]) for ind in items])
			
			root_predicate_start_idx = min(root_verb['verb_idx'])
			root_predicate_end_idx = max(root_verb['verb_idx']) + 1
						
			predicates.append(root_predicate)
			predicates_idx.append((root_predicate_start_idx, root_predicate_end_idx))
			use_predicate_span = True				
			
			#print(root_predicate)
			#print((root_predicate_start_idx, root_predicate_end_idx))
			#input('enter..')
		'''
		
		
		
		total_num_of_samples += 1
		
		# debug
		'''
		if entity_1 not in entity_type_dict:
			print('Error!! entity_1 is not in entity_type_dict:', entity_1)
			input('enter...')
		if entity_2 not in entity_type_dict:
			print('Error!! entity_2 is not in entity_type_dict:', entity_2)
			input('enter...')
		'''
		
		relation = {'rel_id': rel_id, 
					'rel_type': rel_type, 
					'entity_1': entity_1,
					'entity_1_idx': (e1_start, e1_end),
					#'entity_1_type': entity_1_type,
					'entity_1_type': 'gene/protein', # for HPRD50
					'entity_2': entity_2,
					'entity_2_idx': (e2_start, e2_end),
					#'entity_2_type': entity_2_type,
					'entity_2_type': 'gene/protein',  # for HPRD50
					'use_predicate_span': use_predicate_span,
					# 'predicates': [] and 'predicates_idx': [] to avoid an error when running HPRD50 11-27-2021
					# error msg: pyarrow.lib.ArrowNotImplementedError: Unsupported cast from struct<rel_id: int64, rel_type: string, entity_1: string, entity_1_idx: list<item: int64>, entity_1_type: string, entity_2: string, entity_2_idx: list<item: int64>, entity_2_type: string, use_predicate_span: bool, predicates: list<item: null>, predicates_idx: list<item: null>> to struct using function cast_struct
					#'predicates': predicates,
					'predicates': [],
					#'predicates_idx': predicates_idx}
					'predicates_idx': []}
		
		output_txt += json.dumps({"id": sent_id, # TODO: sent_id is not unique. change it to something unique.
								  "tokens": tokens,
								  "relation": [relation],
								  "directed": True, # relation directionality. a.k.a symmetric or asymmetric relation.
								  "reverse": False}) # this is only used for undirected relations. 
												   # For testing phase, undirected samples are replicated, and the replicated samples are tagged as reverse. 
												   # So, if it's set to true, the model uses the second entity + the first entity instead of 
												   # the first entity + the second entity to classify both relation representation cases (A + B, B + A). 
		output_txt += '\n'
		
		
		
		
	outfile = os.path.join(pkl_dir, pkl_file.replace('df_', '').replace('pkl', 'json'))
	with open(outfile, "w") as f:
		f.write(output_txt)