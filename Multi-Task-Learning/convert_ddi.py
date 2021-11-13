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


overlapped_entities_counter = 0

data_dir = 'datasets/ddi/scifive_unreplaced'

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
