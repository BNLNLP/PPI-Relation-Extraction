"""
This code converts the DDI format from SciFive to the format that fits to the model. 11-13-2021

- DDI has a single label for a sentence.
- DDI contains overlapped entities. --> Only 2 cases.
  e.g., The objective of this study was to evaluate the effect of oral administration of ginseng stem-and-leaf saponins (GSLS) on the humoral immune responses of chickens to * ina # ctivated ND # * and AI vaccines.	false


ref: https://github.com/justinphan3110/SciFive/blob/main/finetune/re/ddi_1.ipynb

"""
import os
import csv
import json

import spacy
from spacy.tokens import Doc




# read entity types.
ddi_corpus_dir = 'datasets/ddi/bluebert/DDICorpusBrat/'

entity_type_dict = {}

num_of_splitted_entities = 0 # debug

for root, dirs, files in os.walk(ddi_corpus_dir):
	for file in files:
		
		# debug
		'''
		if file.endswith(".txt"):
			f = open(os.path.join(root, file))
			for line in f.readlines():
				if 'The objective of this study was to evaluate' in line:
					print(os.path.join(root, file))
					input('enter..')
		'''
		
		if file.endswith(".ann"): 
			
			f = open(os.path.join(root, file))
			
			for line in f.readlines():
				#print(line)
				#print(line.split('\t'))
				#input('enter..')
				
				if line.startswith('R'): # skip relations.
					continue

				line = line.split('\t')
				e_type = line[1].split()[0].strip()
				e_name = line[2].strip()
				
				
				# debug
				#if file == '21844260.ann':
				if ';' in line[1]:
					print(line)
					print(e_type, e_name)
					num_of_splitted_entities += 1
					#input('enter..')
				
				
				
				
				if e_name in entity_type_dict:
					"""
					Some entities have different types. I'm not sure if it's an error. In this case, just the most common type.
					e.g., Nrf2 (GENE-Y, GENE-N), matriptase-2 (GENE-Y, GENE-N), EROD (CHEMICAL, GENE-N)
					"""
					if e_type in entity_type_dict[e_name]:
						entity_type_dict[e_name][e_type] += 1
					else:
						entity_type_dict[e_name][e_type] = 1
					'''
					if entity_type_dict[e_name] != e_type:
						print('Error!! - Different type for the same entity.')
						print(e_name, entity_type_dict[e_name], e_type)
						input('enter...')
					'''		
				else:
					entity_type_dict[e_name] = {e_type: 1}

import operator

for k, v in entity_type_dict.items():
	if len(v) == 1:
		entity_type_dict[k] = list(v.keys())[0]
	else:
		entity_type_dict[k] = max(v.items(), key=operator.itemgetter(1))[0]
		# debug
		'''
		for kk, vv in v.items():
			print(k, kk, vv)
		print(max(v.items(), key=operator.itemgetter(1))[0])
		input('enter..')
		'''



nlp = spacy.load('en_core_web_sm')


data_dir = 'datasets/ddi/scifive_unreplaced'


# debug
total_num_of_samples = 0
num_of_samples_with_overlapped_entities = 0
num_of_samples_with_no_context_between_entities = 0
num_of_samples_with_no_predicate_in_context = 0 # count the number of sentences where no verb exists between the two entities.
num_of_samples_with_entity_in_predicate = 0 # count the number of sentences where predicate contains entity because of parsing error.
num_of_predicates_for_samples_with_no_context_between_entities = 0 # count the samples where a predicate having either entity as child exists for samples that no verb exists between the two entities.
num_of_samples_with_error_tags = 0 # count the number of samples where tag error exists such as more than two ' >>'.
num_of_sample_with_split_entities = 0 # count the number smaple having entities composed of separate tokens other than a single span. e.g., T5	DRUG 422 436;444 452	inactivated ND vaccines (from Test/MedLine/21844260.ann)


for filename in os.listdir(data_dir):
	f = open(os.path.join(data_dir, filename))
	
	read_tsv = csv.reader(f, delimiter="\t")
	
	output_txt = ''
	for row in read_tsv:
		text = row[0]
		rel_type = row[1]
		
		
		#if rel_type == 'false': # ignore 'DDI-False' labels.
		#	continue
		
		
		# debug
		#text = 'Changes in urinary homocysteine following * syn # thetic steroidal es # trogen * and progestogen administration to rats.'
		#text = 'The objective of this study was to evaluate the effect of oral administration of ginseng stem-and-leaf saponins (GSLS) on the humoral immune responses of chickens to * ina # ctivated ND # * and AI vaccines.'

		
		# TODO: make it cleaner later. 11-17-2021
		#       currently, it first replaces original markers with new markers and then remove the markers. Remove the markers from original text.
		
		# debug
		'''
		total += 1
		e1_s_t = text.count('* ')
		e1_e_t = text.count(' *')
		e2_s_t = text.count('# ')
		e2_e_t = text.count(' #')
		if e1_s_t != 1 or e1_e_t != 1 or e2_s_t != 1 or e2_e_t != 1:
			print(e1_s_t, e1_e_t, e2_s_t, e2_e_t)
			print(text)
			cnt += 1
			continue
		'''
		
		e1_start_marker_idx = text.index('* ')
		e1_end_marker_idx   = text[e1_start_marker_idx + 2:].index(' *') + e1_start_marker_idx + 2
		e2_start_marker_idx = text.index('# ')
		e2_end_marker_idx   = text[e2_start_marker_idx + 2:].index(' #') + e2_start_marker_idx + 2

		# debug
		if e2_start_marker_idx <= e1_start_marker_idx or e1_start_marker_idx >= e1_end_marker_idx or e2_start_marker_idx >= e2_end_marker_idx:
			print(text)
			print('e1_start_marker_idx:', e1_start_marker_idx, '/ e2_start_marker_idx:', e2_start_marker_idx)
			print('e1_start_marker_idx:', e1_start_marker_idx, '/ e1_end_marker_idx:', e1_end_marker_idx)
			print('e2_start_marker_idx:', e2_start_marker_idx, '/ e2_end_marker_idx:', e2_end_marker_idx)
			input('enter..')
			
		if e2_start_marker_idx >= e1_start_marker_idx and e2_start_marker_idx <= e1_end_marker_idx: # Only 2 cases.
			num_of_samples_with_overlapped_entities += 1
		
		if e1_start_marker_idx >= e2_start_marker_idx and e1_start_marker_idx <= e2_end_marker_idx: # None of this case.
			num_of_samples_with_overlapped_entities += 1
		
		# replace the markers from end so that indices for earlier markers are preserved.
		# - there are overlapped entities. e.g., 'Changes in urinary homocysteine following * syn # thetic steroidal es # trogen * and progestogen administration to rats.'
		l = [e1_start_marker_idx, e1_end_marker_idx, e2_start_marker_idx, e2_end_marker_idx]
		for i in sorted(l, reverse=True):
			if i == e2_end_marker_idx:
				text = text[:e2_end_marker_idx] + '[/E2]' + text[e2_end_marker_idx + 2:]
			elif i == e2_start_marker_idx:
				text = text[:e2_start_marker_idx] + '[E2]' + text[e2_start_marker_idx + 2:]
			elif i == e1_end_marker_idx:
				text = text[:e1_end_marker_idx] + '[/E1]' + text[e1_end_marker_idx + 2:]
			elif i == e1_start_marker_idx:
				text = text[:e1_start_marker_idx] + '[E1]' + text[e1_start_marker_idx + 2:]
		
		# debug
		markers = ['[E1]', '[/E1]', '[E2]', '[/E2]']
		if all(x in text for x in markers) == False:
			print('Error: not all markers are used:', text)
			input('enter...')
			continue

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
				
		adjusted_offset_for_e1_start = 4 # 4 is the length of '[E1]'
		adjusted_offset_for_e1_end   = 5 # 5 is the length of '[/E1]'
		adjusted_offset_for_e2_start = 4 # 4 is the length of '[E2]'
		adjusted_offset_for_e2_end   = 5 # 5 is the length of '[/E1]'
		
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

		l = [e1_start, e1_end, e2_start, e2_end]
		for num, i in enumerate(sorted(l)):
			if i == e1_start:
				if num == 0:
					final_adjusted_offset_for_e1_start = adjusted_offset_for_e1_start
				elif num == 1:
					final_adjusted_offset_for_e1_start = adjusted_offset_for_e1_start + adjusted_offset_for_e2_start
				elif num == 2:
					final_adjusted_offset_for_e1_start = adjusted_offset_for_e1_start + adjusted_offset_for_e2_start + adjusted_offset_for_e2_end
			elif i == e1_end:
				if num == 1:
					final_adjusted_offset_for_e1_end = adjusted_offset_for_e1_start
				elif num == 2:
					final_adjusted_offset_for_e1_end = adjusted_offset_for_e1_start + adjusted_offset_for_e2_start
				elif num == 3:
					final_adjusted_offset_for_e1_end = adjusted_offset_for_e1_start + adjusted_offset_for_e2_start + adjusted_offset_for_e2_end
			elif i == e2_start:
				if num == 0:
					final_adjusted_offset_for_e2_start = adjusted_offset_for_e2_start
				elif num == 1:
					final_adjusted_offset_for_e2_start = adjusted_offset_for_e2_start + adjusted_offset_for_e1_start
				elif num == 2:
					final_adjusted_offset_for_e2_start = adjusted_offset_for_e2_start + adjusted_offset_for_e1_start + adjusted_offset_for_e1_end
			elif i == e2_end:
				if num == 1:
					final_adjusted_offset_for_e2_end = adjusted_offset_for_e2_start
				elif num == 2:
					final_adjusted_offset_for_e2_end = adjusted_offset_for_e2_start + adjusted_offset_for_e1_start
				elif num == 3:
					final_adjusted_offset_for_e2_end = adjusted_offset_for_e2_start + adjusted_offset_for_e1_start + adjusted_offset_for_e1_end
		
		e1_start -= final_adjusted_offset_for_e1_start
		e1_end   -= final_adjusted_offset_for_e1_end
		e2_start -= final_adjusted_offset_for_e2_start
		e2_end   -= final_adjusted_offset_for_e2_end
		
		# debug
		n1 = text[e1_start:e1_end] # debug
		n2 = text[e2_start:e2_end] # debug
		if entity_1 != n1 or entity_2 != n2: # they must be the same.
			print('original text:', row[0])
			print('modified text:', text)
			print('e1_start:', e1_start, '/ e1_end:', e1_end)
			print('e2_start:', e2_start, '/ e2_end:', e2_end)
			print('e1:', entity_1, '/ e2:', entity_2)
			print('n1:', n1, '/ n2:', n2)
			#input('enter...')



		if rel_type == 'false':
			rel_type = 'DDI-false'
		
		# debug
		if rel_type not in ['DDI-false', 'DDI-advise', 'DDI-effect', 'DDI-int', 'DDI-mechanism']:
			print('Error: UNKNOWN relation type - ', rel_type)
			input('enter...')
		
		if rel_type == 'DDI-false':
			rel_id = 0
		elif rel_type == 'DDI-advise':
			rel_id = 1
		elif rel_type == 'DDI-effect':
			rel_id = 2 
		elif rel_type == 'DDI-int':
			rel_id = 3 
		elif rel_type == 'DDI-mechanism':
			rel_id = 4
		
		
		
		
		
		
		
		
		
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
	
		# debug - counting no context (tokens between entities)
		if e2_start == e1_end or e1_start == e2_end:
			num_of_samples_with_no_context_between_entities += 1
		
		# debug - error checking
		if e1_end <= e1_start or e2_end <= e2_start:
			print(text)
			print(tokens)
			print('e1_start:', e1_start, '/ e1_end:', e1_end)
			print('e2_start:', e2_start, '/ e2_end:', e2_end)
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
		
		
		
		# debug
		is_entity_split = False
		if entity_1 not in entity_type_dict:
			found = False
			for k in entity_type_dict.keys():
				if entity_1 in k:
					e1_type = entity_type_dict[k]
					found = True
					'''
					print('k in entity:', k)
					print('entity_type_dict[k]:', entity_type_dict[k])
					input('enter..')
					'''
					break
			
			if found:
				entity_type_dict[entity_1] = e1_type
			else:
				print('Error!! entity_1 is not in entity_type_dict:', entity_1)
				print('text:', row[0])
				is_entity_split = True
				#input('enter...')
		
		if entity_2 not in entity_type_dict:
			found = False
			for k in entity_type_dict.keys():
				if entity_2 in k:
					e2_type = entity_type_dict[k]
					found = True
					'''
					print('k in entity:', k)
					print('entity_type_dict[k]:', entity_type_dict[k])
					input('enter..')
					'''
					break
			
			if found:
				entity_type_dict[entity_2] = e2_type
			else:
				print('Error!! entity_2 is not in entity_type_dict:', entity_2)
				print('text:', row[0])
				is_entity_split = True
				#input('enter...')
			
		if is_entity_split: # skip samples with a split entity.
			num_of_sample_with_split_entities += 1
			continue
		
		total_num_of_samples += 1
		
		relation = {'rel_id': rel_id, 
					'rel_type': rel_type, 
					'entity_1': entity_1,
					'entity_1_idx': (e1_start, e1_end),
					'entity_1_type': entity_type_dict[entity_1],
					'entity_2': entity_2,
					'entity_2_idx': (e2_start, e2_end),
					'entity_2_type': entity_type_dict[entity_2],
					'use_predicate_span': use_predicate_span,
					'predicates': predicates,
					'predicates_idx': predicates_idx}
					
		output_txt += json.dumps({"id": None, # TODO: put any unique id although it's not used.
								  "tokens": tokens,
								  "relation": [relation],
								  "directed": True, # relation directionality. a.k.a symmetric or asymmetric relation.
								  "reverse": False}) # this is only used for undirected relations. 
												   # For testing phase, undirected samples are replicated, and the replicated samples are tagged as reverse. 
												   # So, if it's set to true, the model uses the second entity + the first entity instead of 
												   # the first entity + the second entity to classify both relation representation cases (A + B, B + A). 
		output_txt += '\n'
		
	
	outfile = os.path.join('datasets/ddi/converted', filename.replace('_unreplaced.tsv', '_0.json'))
	with open(outfile, "w") as f:
		f.write(output_txt)

# debug
print('>> total_num_of_samples:', total_num_of_samples)
print('>> num_of_samples_with_overlapped_entities:', num_of_samples_with_overlapped_entities)
print('>> num_of_samples_with_no_context_between_entities:', num_of_samples_with_no_context_between_entities)
print('>> num_of_samples_with_no_predicate_in_context:', num_of_samples_with_no_predicate_in_context)
print('>> num_of_samples_with_entity_in_predicate:', num_of_samples_with_entity_in_predicate)
print('>> num_of_predicates_for_samples_with_no_context_between_entities:', num_of_predicates_for_samples_with_no_context_between_entities)
print('>> num_of_splitted_entities:', num_of_splitted_entities)
print('>> num_of_sample_with_split_entities:', num_of_sample_with_split_entities)



""" deprecated: old way of conversion using 'entity_marked_sent'

for filename in os.listdir(data_dir):
	f = open(os.path.join(data_dir, filename))
	
	read_tsv = csv.reader(f, delimiter="\t")
	
	output_txt = ''
	for row in read_tsv:
		text = row[0]
		rel_type = row[1]

		# debug
		'''
		total += 1
		e1_s_t = text.count('* ')
		e1_e_t = text.count(' *')
		e2_s_t = text.count('# ')
		e2_e_t = text.count(' #')
		if e1_s_t != 1 or e1_e_t != 1 or e2_s_t != 1 or e2_e_t != 1:
			print(e1_s_t, e1_e_t, e2_s_t, e2_e_t)
			print(text)
			cnt += 1
			continue
		'''
		
		e1_start_marker_idx = text.index('* ')
		e1_end_marker_idx   = text[e1_start_marker_idx + 2:].index(' *') + e1_start_marker_idx + 2
		e2_start_marker_idx = text.index('# ')
		e2_end_marker_idx   = text[e2_start_marker_idx + 2:].index(' #') + e2_start_marker_idx + 2
		
		e1_text = text[e1_start_marker_idx + 2:e1_end_marker_idx]
		e2_text = text[e2_start_marker_idx + 2:e2_end_marker_idx]
		
		# debug
		if e2_start_marker_idx <= e1_start_marker_idx or e1_start_marker_idx >= e1_end_marker_idx or e2_start_marker_idx >= e2_end_marker_idx:
			print(text)
			print('e1_start_marker_idx:', e1_start_marker_idx, '/ e2_start_marker_idx:', e2_start_marker_idx)
			print('e1_start_marker_idx:', e1_start_marker_idx, '/ e1_end_marker_idx:', e1_end_marker_idx)
			print('e2_start_marker_idx:', e2_start_marker_idx, '/ e2_end_marker_idx:', e2_end_marker_idx)
			input('enter..')
			
		if e2_start_marker_idx >= e1_start_marker_idx and e2_start_marker_idx <= e1_end_marker_idx: # Only 2 cases.
			overlapped_entities_counter += 1
		
		if e1_start_marker_idx >= e2_start_marker_idx and e1_start_marker_idx <= e2_end_marker_idx: # None of this case.
			overlapped_entities_counter += 1
		
		# replace the markers from end so that indices for earlier markers are preserved.
		text_with_new_marker = text[:e2_end_marker_idx] + '[/E2]' + text[e2_end_marker_idx + 2:]
		text_with_new_marker = text_with_new_marker[:e2_start_marker_idx] + '[E2]' + text_with_new_marker[e2_start_marker_idx + 2:]
		text_with_new_marker = text_with_new_marker[:e1_end_marker_idx] + '[/E1]' + text_with_new_marker[e1_end_marker_idx + 2:]
		text_with_new_marker = text_with_new_marker[:e1_start_marker_idx] + '[E1]' + text_with_new_marker[e1_start_marker_idx + 2:]
		
		# debug
		markers = ['[E1]', '[/E1]', '[E2]', '[/E2]']
		if all(x in text_with_new_marker for x in markers) == False:
			print('Error: not all markers are used:', text_with_new_marker)
			input('enter...')
			continue
			
		
		#if text.startswith('* Antihistamines * : #'):
		#	print(text)
		#	print(text_with_new_marker)
		#	input('enter..')
	
	
		if rel_type == 'false':
			rel_type = 'DDI-false'
		
		# debug
		if rel_type not in ['DDI-false', 'DDI-advise', 'DDI-effect', 'DDI-int', 'DDI-mechanism']:
			print('Error: UNKNOWN relation type - ', rel_type)
			input('enter...')
		
		if rel_type == 'DDI-false':
			rel_id = 0
		elif rel_type == 'DDI-advise':
			rel_id = 1
		elif rel_type == 'DDI-effect':
			rel_id = 2 
		elif rel_type == 'DDI-int':
			rel_id = 3 
		elif rel_type == 'DDI-mechanism':
			rel_id = 4
			
		relation = {'rel_id': rel_id,
					'rel_type': rel_type, 
					'entity_1': e1_text,
					'entity_2': e2_text}
		
		output_txt += json.dumps({"id": None, # TODO: put any unique id although it's not used.
								  "entity_marked_sent": text_with_new_marker,
								  "relation": [relation],
								  "directed": True, # relation directionality. a.k.a symmetric or asymmetric relation.
								  "reverse": False}) # this is only used for undirected relations. 
												   # For testing phase, undirected samples are replicated, and the replicated samples are tagged as reverse. 
												   # So, if it's set to true, the model uses the second entity + the first entity instead of 
												   # the first entity + the second entity to classify both relation representation cases (A + B, B + A). 
		output_txt += '\n'
		
	
	outfile = os.path.join('datasets/ddi/converted', filename.replace('_unreplaced.tsv', '_0.json'))
	with open(outfile, "w") as f:
		f.write(output_txt)

print('>> overlapped_entities_counter:', overlapped_entities_counter)
"""