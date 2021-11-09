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
			
			relation_id = 0 # ADE has a single class.
			
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
			
	data_type = file.rsplit('_', 1)[1].replace('.json', '')
	data_num = file.rsplit('_', 2)[1]

	outfile = os.path.join('datasets/ade/converted', data_type + '_' + data_num + '.json')
	with open(outfile, "w") as f:
		f.write(output_txt)
