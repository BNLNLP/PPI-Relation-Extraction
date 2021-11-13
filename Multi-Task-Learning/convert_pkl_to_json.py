"""
This code converts the old data format (pickle for DataFrame) to the JSON format. 11-04-2021

"""
import os
import pickle
import pandas as pd
import json


pkl_dir = 'datasets/ppi/original/LLL/all_ver_5'
#pkl_dir = 'datasets/ppi/type_annotation/ALL/all_incl_negative_annotation_ver_17'

for pkl_file in os.listdir(pkl_dir):
	if not pkl_file.startswith('df_'): # skip relations files.
		continue
		
	df = pd.read_pickle(os.path.join(pkl_dir, pkl_file))

	json_txt = ''
	for idx, row in df.iterrows():		
		entity_marked_sent = row['sents']
		relation = row['relations'].replace('(e1,e2)', '')
		relation_id = row['relations_id']
		sent_id = row['sent_ids']
		
		json_txt += json.dumps({"pair_id": sent_id, # pair_id was not stored in the DataFrame, so just put in sent_id instead.
								"sent_id": sent_id,
								"entity_marked_sent": entity_marked_sent,
								"relation": relation,
								"relation_id": relation_id,
								"directed": False, # relation directionality. a.k.a symmetric or asymmetric relation.
								"reverse": False}) # this is only used for undirected relations. 
												   # For testing phase, undirected samples are replicated, and the replicated samples are tagged as reverse. 
												   # So, if it's set to true, the model uses the second entity + the first entity instead of 
												   # the first entity + the second entity to classify both relation representation cases (A + B, B + A). 

		json_txt += '\n'
		
		'''
		# For evaluation and prediction, add a reverse relations sample if it's undirectional so that the model classify both relation representation cases.
		# For training, the reverse relation representation is generated for loss calculation, but this can't be used for testing since each sample is assigned to a single label. 
		if pkl_file.startswith('df_dev') or pkl_file.startswith('df_test'):
			json_txt += json.dumps({"pair_id": sent_id,
									"sent_id": sent_id,
									"entity_marked_sent": entity_marked_sent,
									"relation": relation,
									"relation_id": relation_id,
									"directed": False,
									"reverse": True}) 
			json_txt += '\n'
		'''
		
	outfile = os.path.join(pkl_dir, pkl_file.replace('df_', '').replace('pkl', 'json'))
	with open(outfile, "w") as f:
		f.write(json_txt)