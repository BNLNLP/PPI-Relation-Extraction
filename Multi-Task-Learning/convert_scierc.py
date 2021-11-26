"""
This code converts the SciERC format from SpERT to the format that fits to the model. 11-08-2021

- SciERC data contains overlapped entities.
- SciERC data contains samples with no relations.

"""
import os
import json

import spacy
from spacy.tokens import Doc

nlp = spacy.load('en_core_web_sm')

def custom_tokenizer(tokens): # the text is already tokenized.
    return Doc(nlp.vocab, tokens)

nlp.tokenizer = custom_tokenizer
	

# debug
total_num_of_samples = 0
num_of_samples_with_overlapped_entities = 0
num_of_samples_with_no_context_between_entities = 0
num_of_samples_with_no_predicate_in_context = 0 # count the number of sentences where no verb exists between the two entities.
num_of_samples_with_entity_in_predicate = 0 # count the number of sentences where predicate contains entity because of parsing error.
# TODO: rename this variable.
num_of_predicates_for_samples_with_no_context_between_entities = 0 # count the samples where a predicate having either entity as child exists for samples that no verb exists between the two entities.


data_dir = 'datasets/scierc/spert'

for file in os.listdir(data_dir):
	if file not in ['scierc_train.json', 'scierc_test.json']:
		continue

	data = json.load(open(os.path.join(data_dir, file)))
	
	output_txt = ''
	for item in data:
		tokens = item['tokens']
		entities = item['entities']
		relations = item['relations']
		orig_id = item['orig_id']
		
		'''
		if '◆' in tokens:
			idx = tokens.index('◆')
			tokens[idx]  = '-'
		'''
		
		
		# debug
		'''
		tokens = ["Amorph", "recognizes", "NE", "items", "in", "two", "stages", ":", "dictionary", "lookup", "and", "rule", "application", "."]
		entities= [{
				"type": "Method",
				"start": 0,
				"end": 1
			}, {
				"type": "OtherScientificTerm",
				"start": 2,
				"end": 4
			}, {
				"type": "Method",
				"start": 8,
				"end": 10
			}, {
				"type": "Method",
				"start": 11,
				"end": 13
			}
		]
		relations = [{
				"type": "Used-for",
				"head": 0,
				"tail": 1
			}, {
				"type": "Part-of",
				"head": 2,
				"tail": 0
			}, {
				"type": "Conjunction",
				"head": 2,
				"tail": 3
			}, {
				"type": "Part-of",
				"head": 3,
				"tail": 0
			}
		]
		'''
		

		rel_list = []
		for rel in relations:
			rel_type = rel['type']
			head = rel['head']
			tail = rel['tail']
			
			e1_start = entities[head]['start']
			e1_end = entities[head]['end']
			e2_start = entities[tail]['start']
			e2_end = entities[tail]['end']
			
			entity_1 = tokens[e1_start:e1_end]
			entity_2 = tokens[e2_start:e2_end]
			
			entity_1_type = entities[head]['type']
			entity_2_type = entities[tail]['type']
			
			if rel_type == 'Used-for':
				rel_id = 0
			elif rel_type == 'Feature-of':
				rel_id = 1
			elif rel_type == 'Hyponym-of':
				rel_id = 2 
			elif rel_type == 'Evaluate-for':
				rel_id = 3 
			elif rel_type == 'Part-of':
				rel_id = 4
			elif rel_type == 'Compare':
				rel_id = 5
			elif rel_type == 'Conjunction':
				rel_id = 6
			

			
			'''
			print('>> tokens:', tokens)
			print('>> entities:', entity_1, '(', e1_start, ',', e1_end, ')', entity_2, '(', e2_start, ',', e2_end, ')')
			print('>> rel_type:', rel_type)
			'''
			
			# debug
			#tokens = ['he', 'did', 'not', 'want', 'to', 'swim']
			
			
			doc = nlp(tokens)
			
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
			
			for idx, token in enumerate(doc):

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
					
					
					
					
				'''	
				for child in doc[token.i].children:
					if (child.i >= e1_start and child.i < e1_end) or (child.i >= e2_start and child.i < e2_end):
						print('>> token: ' + token + ', token.pos_:' + token.pos_)
						print('>> token.children:', [x.text for x in doc[token.i].children])
						print('>> child:', child, '(' + str(child.i) + ')')
						print('>> tokens:', tokens)
						print('>> entities:', entity_1, '(' + str(e1_start) + ', '+ str(e1_end) + ')', entity_2, '(' + str(e2_start) + ', '+ str(e2_end) + ')')
						print('>> rel_type:', rel_type)
						input('enter..')
				'''
				
				'''
				is_e1_child = False
				is_e2_child = False
				
				for child in doc[token.i].children:
					if (child.i >= e1_start and child.i < e1_end):
						is_e1_child = True
					if (child.i >= e2_start and child.i < e2_end):
						is_e2_child = True
						
				if is_e1_child and is_e2_child:
					print('>> token: ', token, ', token.pos_:', token.pos_)
					print('>> token.children:', [x.text for x in doc[token.i].children])
					#print('>> child:', child, '(' + str(child.i) + ')')
					print('>> tokens:', tokens)
					print('>> entities:', entity_1, '(' + str(e1_start) + ', '+ str(e1_end) + ')', entity_2, '(' + str(e2_start) + ', '+ str(e2_end) + ')')
					print('>> rel_type:', rel_type)
					input('enter..')
				'''	

			
			# debug - error checking
			if e1_end <= e1_start or e2_end <= e2_start:
				print('tokens:', tokens)
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
			'''
			print('rel_id:', rel_id)
			print('rel_type:', rel_type)
			print('entity_1:', entity_1)
			print('entity_1_idx:', (e1_start, e1_end))
			print('entity_2:', entity_2)
			print('entity_2_idx:', (e2_start, e2_end))
			print('use_predicate_span:', use_predicate_span)
			print('predicates:', predicates)
			print('predicates_idx:', predicates_idx)
			input('enter..')
			'''
		
			total_num_of_samples += 1

			rel_list.append({'rel_id': rel_id, 
							 'rel_type': rel_type, 
							 'entity_1': entity_1,
							 'entity_1_idx': (e1_start, e1_end),
							 'entity_1_type': entity_1_type,
							 'entity_2': entity_2,
							 'entity_2_idx': (e2_start, e2_end),
							 'entity_2_type': entity_2_type,
							 'use_predicate_span': use_predicate_span,
							 'predicates': predicates,
							 'predicates_idx': predicates_idx})

		# skip samples with no relations.
		if len(rel_list) == 0:
			continue

		output_txt += json.dumps({"id": orig_id,
								  "tokens": tokens,
								  "relation": rel_list,
								  "directed": True, # relation directionality. a.k.a symmetric or asymmetric relation.
								  "reverse": False}) # this is only used for undirected relations. 
												   # For testing phase, undirected samples are replicated, and the replicated samples are tagged as reverse. 
												   # So, if it's set to true, the model uses the second entity + the first entity instead of 
												   # the first entity + the second entity to classify both relation representation cases (A + B, B + A). 
		output_txt += '\n'
	
	outfile = ''
	if file == 'scierc_train.json':
		outfile = 'train_0.json'
	elif file == 'scierc_test.json':
		outfile = 'test_0.json'
		
	outfile = os.path.join('datasets/scierc/converted', outfile)

	with open(outfile, "w") as f:
		f.write(output_txt)


# debug
print('>> total_num_of_samples:', total_num_of_samples)
print('>> num_of_samples_with_overlapped_entities:', num_of_samples_with_overlapped_entities)
print('>> num_of_samples_with_no_context_between_entities:', num_of_samples_with_no_context_between_entities)
print('>> num_of_samples_with_no_predicate_in_context:', num_of_samples_with_no_predicate_in_context)
print('>> num_of_samples_with_entity_in_predicate:', num_of_samples_with_entity_in_predicate)
print('>> num_of_predicates_for_samples_with_no_context_between_entities:', num_of_predicates_for_samples_with_no_context_between_entities)

			

