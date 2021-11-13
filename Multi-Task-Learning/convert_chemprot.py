"""
This code converts the CHEMPROT format from SciBERT/SciFive to the format that fits to the model. 11-12-2021

- CHEMPROT has a single label for a sentence.

ref: https://github.com/allenai/scibert/blob/master/scripts/chemprot_to_relation.py

"""
import os
import csv
import json


data_dir = 'datasets/chemprot/scifive_txt'


# debug
total = 0
cnt = 0
	
	
for filename in os.listdir(data_dir):

	f = open(os.path.join(data_dir, filename))
	
	'''
	read_tsv = csv.reader(f, delimiter="\t")

	for row in read_tsv:
		print(row)
		text = row[0]
		
		e1 = text.count('**')
		e2 = text.count('##')
		
		# debug
		if e1 != 2 or e2 != 2:
			print(text)
			print(e1, e2)
			input('enter..')
	
	input('enter.................')
	'''
	
	output_txt = ''
	for line in f.readlines():
		line = json.loads(line)
		
		text = line['text']
		rel_type = line['label']

		# <<, >>, [[, ]]

		# debug
		total += 1
		e1_s_t = text.count('<< ')
		e1_e_t = text.count(' >>')
		e2_s_t = text.count('[[ ')
		e2_e_t = text.count(' ]]')
		if e1_s_t != 1 or e1_e_t != 1 or e2_s_t != 1 or e2_e_t != 1:
			print(e1_s_t, e1_e_t, e2_s_t, e2_e_t)
			print(text)
			cnt += 1
			continue

		e1_start = text.index('<< ') + 3
		e1_end = text.index(' >>')
		e2_start = text.index('[[ ') + 3
		e2_end = text.index(' ]]')
		
		e1_text = text[e1_start:e1_end]
		e2_text = text[e2_start:e2_end]

		text_with_new_marker = text.replace('<< ', '[E1]').replace(' >>', '[/E1]').replace('[[ ', '[E2]').replace(' ]]', '[/E2]')
		
		if rel_type not in ['AGONIST-ACTIVATOR', 'DOWNREGULATOR', 'SUBSTRATE_PRODUCT-OF', 'AGONIST', 'INHIBITOR', 'PRODUCT-OF', 
						    'ANTAGONIST', 'ACTIVATOR', 'INDIRECT-UPREGULATOR', 'SUBSTRATE', 'INDIRECT-DOWNREGULATOR', 'AGONIST-INHIBITOR', 
						    'UPREGULATOR', ]:
			print('Error: UNKNOWN relation type - ', rel_type)
			input('enter...')
		
		if rel_type == 'AGONIST-ACTIVATOR':
			rel_id = 0
		elif rel_type == 'DOWNREGULATOR':
			rel_id = 1
		elif rel_type == 'SUBSTRATE_PRODUCT-OF':
			rel_id = 2 
		elif rel_type == 'AGONIST':
			rel_id = 3 
		elif rel_type == 'INHIBITOR':
			rel_id = 4
		elif rel_type == 'PRODUCT-OF':
			rel_id = 5
		elif rel_type == 'ANTAGONIST':
			rel_id = 6
		elif rel_type == 'ACTIVATOR':
			rel_id = 7
		elif rel_type == 'INDIRECT-UPREGULATOR':
			rel_id = 8
		elif rel_type == 'SUBSTRATE':
			rel_id = 9
		elif rel_type == 'INDIRECT-DOWNREGULATOR':
			rel_id = 10
		elif rel_type == 'AGONIST-INHIBITOR':
			rel_id = 11
		elif rel_type == 'UPREGULATOR':
			rel_id = 12
			
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


	outfile = os.path.join('datasets/chemprot/converted', filename.replace('.txt', '_0.json'))
	with open(outfile, "w") as f:
		f.write(output_txt)

# debug
print(total)
print(cnt)
	
	