"""
This code converts the CoNLL04 format from SpERT to the format that fits to the model. 11-08-2021

- CoNLL04 data doesn't contains overlapped entities.

"""
import os
import json


data_dir = 'datasets/conll04/spert'

for file in os.listdir(data_dir):
	if file not in ['conll04_train.json', 'conll04_test.json']:
		continue

	data = json.load(open(os.path.join(data_dir, file)))
	
	output_txt = ''
	for item in data:
		tokens = item['tokens']
		entities = item['entities']
		relations = item['relations']
		orig_id = item['orig_id']
		
		rel_list = []
		for rel in relations:
			rel_type = rel['type']
			head = rel['head']
			tail = rel['tail']
			
			e1_start = entities[head]['start']
			e1_end = entities[head]['end']
			e2_start = entities[tail]['start']
			e2_end = entities[tail]['end']

			# debug - checking overlapped entities (overlapped entities exist!!)
			'''
			if (e1_start >= e2_start and e1_start <= e2_end) or (e2_start >= e1_start and e2_start <= e1_end):
				print(rel)
				input('enter...')
			'''

			if rel_type == 'Work_For':
				rel_id = 0
			elif rel_type == 'Kill':
				rel_id = 1
			elif rel_type == 'OrgBased_In':
				rel_id = 2 
			elif rel_type == 'Live_In':
				rel_id = 3 
			elif rel_type == 'Located_In':
				rel_id = 4
				
			rel_list.append({'rel_id': rel_id, 
							 'rel_type': rel_type, 
							 'entity_1': tokens[e1_start:e1_end],
							 'entity_1_idx': (e1_start, e1_end),
							 'entity_2': tokens[e2_start:e2_end],
							 'entity_2_idx': (e2_start, e2_end)})

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
	if file == 'conll04_train.json':
		outfile = 'train_0.json'
	elif file == 'conll04_test.json':
		outfile = 'test_0.json'
		
	outfile = os.path.join('datasets/conll04/converted', outfile)

	with open(outfile, "w") as f:
		f.write(output_txt)


"""
deprecated - old way of conversion

for file in os.listdir(data_dir):
	if file not in ['conll04_train.json', 'conll04_test.json']:
		continue
		
	data = json.load(open(os.path.join(data_dir, file)))
	
	output_txt = ''
	for item in data:
		tokens = item['tokens']
		entities = item['entities']
		relations = item['relations']
		orig_id = item['orig_id']
		
		
		for num, rel in enumerate(relations, start=1):
			type = rel['type']
			head = rel['head']
			tail = rel['tail']
			
			e1_start = entities[head]['start']
			e1_end = entities[head]['end']
			e2_start = entities[tail]['start']
			e2_end = entities[tail]['end']
			
			entity_marked_sent = tokens.copy()
			
			entity_marked_sent[e1_start] = '[E1]' + entity_marked_sent[e1_start]
			entity_marked_sent[e1_end-1] = entity_marked_sent[e1_end-1] + '[/E1]'
			
			entity_marked_sent[e2_start] = '[E2]' + entity_marked_sent[e2_start]
			entity_marked_sent[e2_end-1] = entity_marked_sent[e2_end-1] + '[/E2]'

			# debug - checking overlapped entities (overlapped entities exist!!)
			'''
			if (e1_start >= e2_start and e1_start <= e2_end) or (e2_start >= e1_start and e2_start <= e1_end):
				print(rel)
				input('enter...')
			'''

			if type == 'Work_For':
				relation_id = 0
			elif type == 'Kill':
				relation_id = 1
			elif type == 'OrgBased_In':
				relation_id = 2 
			elif type == 'Live_In':
				relation_id = 3 
			elif type == 'Located_In':
				relation_id = 4
			
			output_txt += json.dumps({"pair_id": str(orig_id) + '_' + str(num),
									  "sent_id": str(orig_id),
									  "entity_marked_sent": ' '.join(entity_marked_sent),
									  "relation": type,
									  "relation_id": relation_id,
									  "directed": True, # relation directionality. a.k.a symmetric or asymmetric relation.
									  "reverse": False}) # this is only used for undirected relations. 
													   # For testing phase, undirected samples are replicated, and the replicated samples are tagged as reverse. 
													   # So, if it's set to true, the model uses the second entity + the first entity instead of 
													   # the first entity + the second entity to classify both relation representation cases (A + B, B + A). 
			output_txt += '\n'
	
	outfile = ''
	if file == 'conll04_train.json':
		outfile = 'train_0.json'
	elif file == 'conll04_test.json':
		outfile = 'test_0.json'
		
	outfile = os.path.join('datasets/conll04/converted', outfile)

	with open(outfile, "w") as f:
		f.write(output_txt)
"""
