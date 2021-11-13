"""
This code converts the ADE format from SpERT to the format that fits to the model. 11-08-2021

- ADE data contains overlapped entities.

"""
import os
import json


data_dir = 'datasets/ade/spert'

for file in os.listdir(data_dir):
	if not file.startswith('ade_split'):
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
			
			rel_id = 0 # ADE has a single class.
			
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
			
	data_type = file.rsplit('_', 1)[1].replace('.json', '')
	data_num = file.rsplit('_', 2)[1]

	outfile = os.path.join('datasets/ade/converted', data_type + '_' + data_num + '.json')
	with open(outfile, "w") as f:
		f.write(output_txt)
