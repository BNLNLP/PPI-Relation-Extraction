import os
import sys
import csv
import re 
import pandas as pd
import numpy as np
import pickle



output_dir = '/direct/sdcc+u/gpark/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/type_annotation/ALL/all_incl_negative_annotation_ver_19/'

for x in range(10):
	train = pd.read_pickle(output_dir + 'df_train_' + str(x) + '.pkl')
	test = pd.read_pickle(output_dir + 'df_test_' + str(x) + '.pkl')
	
	train_rel_id = {}
	for idx, row in train.iterrows():
		rel_id = row['relations_id']
		if rel_id in train_rel_id:
			train_rel_id[rel_id] += 1
		else:
			train_rel_id[rel_id] = 1
		
	test_rel_id = {}
	for idx, row in test.iterrows():
		rel_id = row['relations_id']
		if rel_id in test_rel_id:
			test_rel_id[rel_id] += 1
		else:
			test_rel_id[rel_id] = 1
			
	for k, v in train_rel_id.items():
		print(k, v)
		
	for k, v in test_rel_id.items():
		print(k, v)	


sys.exit()


output_dir = '/direct/sdcc+u/gpark/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/type_annotation/ALL/all_incl_negative_annotation_ver_21/'
if not os.path.exists(output_dir):
	os.makedirs(output_dir)

num_of_folds = 10
use_dev = False

sents = [] # debug
dup_cnt = 0 # debug
label_dict = {} # debug

for x in range(num_of_folds):
	biocreative_train = pd.read_pickle('/direct/sdcc+u/gpark/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/type_annotation/BioCreative_type/01_27_2021/ver_13/all/df_train_' + str(x) + '.pkl')
	biocreative_test = pd.read_pickle('/direct/sdcc+u/gpark/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/type_annotation/BioCreative_type/01_27_2021/ver_13/all/df_test_' + str(x) + '.pkl')

	aimed_train = pd.read_pickle('/direct/sdcc+u/gpark/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/type_annotation/AImed_type/ver_15/all_no_negative/df_train_' + str(x) + '.pkl')
	aimed_test = pd.read_pickle('/direct/sdcc+u/gpark/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/type_annotation/AImed_type/ver_15/all_no_negative/df_test_' + str(x) + '.pkl')
	#if use_dev:
	#	aimed_dev = pd.read_pickle('/direct/sdcc+u/gpark/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/type_annotation/AImed_type/all/df_dev_' + str(x) + '.pkl')
	
	aimed_neg_train = pd.read_pickle('/direct/sdcc+u/gpark/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/type_annotation/AImed_type/ver_15/all_negative/df_train_' + str(x) + '.pkl')
	aimed_neg_test = pd.read_pickle('/direct/sdcc+u/gpark/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/type_annotation/AImed_type/ver_15/all_negative/df_test_' + str(x) + '.pkl')
	
	bioinfer_train = pd.read_pickle('/direct/sdcc+u/gpark/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/type_annotation/BioInfer_type/ver_15/all_no_negative/df_train_' + str(x) + '.pkl')
	bioinfer_test = pd.read_pickle('/direct/sdcc+u/gpark/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/type_annotation/BioInfer_type/ver_15/all_no_negative/df_test_' + str(x) + '.pkl')
	#if use_dev:
	#	bioinfer_dev = pd.read_pickle('/direct/sdcc+u/gpark/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/type_annotation/BioInfer_type/all/df_dev_' + str(x) + '.pkl')
	
	bioinfer_neg_train = pd.read_pickle('/direct/sdcc+u/gpark/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/type_annotation/BioInfer_type/ver_15/all_negative/df_train_' + str(x) + '.pkl')
	bioinfer_neg_test = pd.read_pickle('/direct/sdcc+u/gpark/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/type_annotation/BioInfer_type/ver_15/all_negative/df_test_' + str(x) + '.pkl')
	
	hprd50_train = pd.read_pickle('/direct/sdcc+u/gpark/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/type_annotation/HPRD50_type/ver_15/all_no_negative/df_train_' + str(x) + '.pkl')
	hprd50_test = pd.read_pickle('/direct/sdcc+u/gpark/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/type_annotation/HPRD50_type/ver_15/all_no_negative/df_test_' + str(x) + '.pkl')
	#if use_dev:
	#	hprd50_dev = pd.read_pickle('/direct/sdcc+u/gpark/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/type_annotation/HPRD50_type/all/df_dev_' + str(x) + '.pkl')
	
	hprd50_neg_train = pd.read_pickle('/direct/sdcc+u/gpark/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/type_annotation/HPRD50_type/ver_15/all_negative/df_train_' + str(x) + '.pkl')
	hprd50_neg_test = pd.read_pickle('/direct/sdcc+u/gpark/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/type_annotation/HPRD50_type/ver_15/all_negative/df_test_' + str(x) + '.pkl')
	
	iepa_train = pd.read_pickle('/direct/sdcc+u/gpark/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/type_annotation/IEPA_type/ver_15/all_no_negative/df_train_' + str(x) + '.pkl')
	iepa_test = pd.read_pickle('/direct/sdcc+u/gpark/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/type_annotation/IEPA_type/ver_15/all_no_negative/df_test_' + str(x) + '.pkl')
	#if use_dev:
	#	iepa_dev = pd.read_pickle('/direct/sdcc+u/gpark/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/type_annotation/IEPA_type/all/df_dev_' + str(x) + '.pkl')
	
	iepa_neg_train = pd.read_pickle('/direct/sdcc+u/gpark/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/type_annotation/IEPA_type/ver_15/all_negative/df_train_' + str(x) + '.pkl')
	iepa_neg_test = pd.read_pickle('/direct/sdcc+u/gpark/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/type_annotation/IEPA_type/ver_15/all_negative/df_test_' + str(x) + '.pkl')
	
	lll_train = pd.read_pickle('/direct/sdcc+u/gpark/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/type_annotation/LLL_type/ver_15/all_no_negative/df_train_' + str(x) + '.pkl')
	lll_test = pd.read_pickle('/direct/sdcc+u/gpark/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/type_annotation/LLL_type/ver_15/all_no_negative/df_test_' + str(x) + '.pkl')
	#if use_dev:
	#	lll_dev = pd.read_pickle('/direct/sdcc+u/gpark/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/type_annotation/LLL_type/all/df_dev_' + str(x) + '.pkl')
	
	train_frames = [biocreative_train, aimed_train, aimed_neg_train, bioinfer_train, bioinfer_neg_train, hprd50_train, hprd50_neg_train, iepa_train, iepa_neg_train, lll_train]
	train = pd.concat(train_frames)

	test_frames = [biocreative_test, aimed_test, aimed_neg_test, bioinfer_test, bioinfer_neg_test, hprd50_test, hprd50_neg_test, iepa_test, iepa_neg_test, lll_test]
	test = pd.concat(test_frames)

	#if use_dev:
	#	dev_frames = [biocreative_dev, aimed_dev, bioinfer_dev, hprd50_dev, iepa_dev, lll_dev]
	#	dev = pd.concat(dev_frames)
	
	
	# debug
	'''
	print(biocreative_train.shape[0])
	print(biocreative_test.shape[0])
	print(biocreative_train.shape[0] + biocreative_test.shape[0])

	input('enter..')
	'''
	
	
	pickle.dump(train, open(os.path.join(output_dir, 'df_train_' + str(x) + '.pkl'), "wb"))
	pickle.dump(test, open(os.path.join(output_dir, 'df_test_' + str(x) + '.pkl'), "wb"))	
	#if use_dev:
	#	pickle.dump(dev, open(os.path.join(output_dir, 'df_dev_' + str(x) + '.pkl'), "wb"))
	
	
	# old data - train-dev-test--70-15-15 (doc based split)
	#test = pd.read_pickle('/direct/sdcc+u/gpark/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/type_annotation/ALL/old - train-dev-test--70-15-15 (doc based split)/all/df_test_' + str(x) + '.pkl')
	
	# debug

	for idx, row in test.iterrows():
		sent = row['sents']
		rel_id = row['relations_id']

		e1_s_idx = sent.index('[E1]')
		e1_e_idx = sent.index('[/E1]')
		e2_s_idx = sent.index('[E2]')
		e2_e_idx = sent.index('[/E2]')
		
		# check if there are any sentences having overlapped entities.
		if (e1_s_idx > e2_s_idx and e1_s_idx < e2_e_idx) or (e2_s_idx > e1_s_idx and e2_s_idx < e1_e_idx):
			print(sent)
			input('enter..')
		
		'''
		if sent in sents:
			# Aimed and BioInfer have three of the same sentences (total 18 samples are overlapped). typed annotation are identical, so just use all of them. 4/21/2021 
			#
			#	a. "Finally, we used in vitro translated proteins in an immunoprecipitation assay to show that, like beta 1-syntrophin, both beta 2- and alpha 1-syntrophin interact with peptides encoding the syntrophin-binding region of dystrophin, utrophin/dystrophin related protein, and the Torpedo 87K protein."
			#	b. "In this recombinant expression system, the dystrophin relatives, human dystrophin related protein (DRP or utrophin) and the 87K postsynaptic protein from Torpedo electric organ, also bind to translated beta 1-syntrophin."
			#	c. "Associations of UBE2I with RAD52, UBL1, p53, and RAD51 proteins in a yeast two-hybrid system."
			print('<test data> data_num:', x)
			print('<test data> sent:', sent)
			print('<test data> relations:', row['relations'])
			print('<test data> relations_id:', row['relations_id'])
			input('enter...')
			dup_cnt += 1
		else:	
			sents.append(row['sents'])
		'''
		
		if rel_id in label_dict:
			label_dict[rel_id] += 1
		else:
			label_dict[rel_id] = 1
	
	
for k, v in label_dict.items():
	print(k, v)
	
print('len(sents):', len(sents))
print('dup_cnt:', dup_cnt)

sys.exit()




















































































































































































































































































'''
bioinfer_train = pd.read_pickle('/direct/sdcc+u/gpark/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/type_annotation/BioInfer_type/all/df_train_' + str(x) + '.pkl')
bioinfer_test = pd.read_pickle('/direct/sdcc+u/gpark/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/type_annotation/BioInfer_type/all/df_test_' + str(x) + '.pkl')

rel = pd.read_pickle('/direct/sdcc+u/gpark/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/type_annotation/BioCreative_type/01_27_2021/all/relations_' + str(x) + '.pkl')

print(rel.rel2idx)
print(rel.idx2rel)

for idx, row in bioinfer_train.iterrows():
	
	print(row)
	print(row['sents'])
	input('enter..')
	
	if not ("[E1]" or "[E2]") in row['sents']:
		print('train')
		print(row['sents'])
		input('enter..')
		
	if row['sents'] in sents:
		print(x, '<train> duplicate sent:', row['sents'])
		input('enter...')
	else:
		sents.append(row['sents'])

for idx, row in bioinfer_test.iterrows():
	if not ("[E1]" or "[E2]") in row['sents']:
		print('dev')
		print(row['sents'])
		input('enter..')

	if row['sents'] in sents:
		print(x, '<dev> duplicate sent:', row['sents'])
		input('enter...')
	else:
		sents.append(row['sents'])

for idx, row in biocreative_test.iterrows():
	if not ("[E1]" or "[E2]") in row['sents']:
		print('test')
		print(row['sents'])
		input('enter..')
	
	if row['sents'] in sents:
		print(x, '<test> duplicate sent:', row['sents'])
		input('enter...')
	else:
		sents.append(row['sents'])

print(set(biocreative_train['relations_id']))
print(set(aimed_train['relations_id']))
print(set(bioinfer_train['relations_id']))
print(set(lll_train['relations_id']))

input('enter..')

print('len(train):', len(train))
print('len(dev):', len(dev))
print('len(test):', len(test))

rel = pd.read_pickle('/direct/sdcc+u/gpark/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/type_annotation/BioInfer_type/all/relations_' + str(x) + '.pkl')

if len(rel.rel2idx) < 3:
	print('rel does not have three classes.')
	input('enter...')
'''




### df_viewer_for_debugging.py

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)


#df = pd.read_pickle('11_26_2020/df_test_0.pkl')
#df = pd.read_pickle('12_22_2020/df_test_0.pkl')
#df = pd.read_pickle('01_04_2021_with_biocreative_test_set/df_test.pkl')

#print(df)

#df = pd.read_pickle('11_26_2020/df_train_0.pkl')
#df = pd.read_pickle('12_22_2020/df_train_0.pkl')
df = pd.read_pickle('01_04_2021_with_biocreative_test_set/df_train.pkl')

print(df)

#rel = pickle.load(open("12_07_2020/relations_0.pkl", "rb"))
#print(type(rel))

### df_viewer_for_debugging.py



#############################################
d = pickle.load( open( "../D.pkl", "rb" ) )

for i in d:
	print(i)
	input('enter...')
	

file = '/direct/sdcc+u/gpark/BER-NLP/BERT-Relation-Extraction/data/BioCreative/sean_annotation_biocreative_training_set.csv'

text = []
with open(file) as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	for row in csv_reader:
		num = row[0].strip()
		sentence = row[2].strip()
		relation = row[3].strip()
		relation_type = row[4].strip()
		comment = row[5].strip()
		
		if len(relation) == 0:
			continue
			
		relation = relation.split(';')
		rel_pairs = []
		for rel in relation:
			print('rel:', rel)
			
			entities = re.split(' -> | - | \? ', rel)
			entities = [x.strip() for x in entities]
			entities = [x.replace('_', ' ') for x in entities]
			
			print('entities:', entities)
			
			
			if len(entities) != 2:
				print('this is not a pair relation:', entities)
				#input()
				continue
			
			e1 = entities[0]
			e2 = entities[1]
				
			if e1 not in sentence or e2 not in sentence:
				print('not existence error e1:', e1)
				print('not existence error e2:', e2)
				print(sentence)
				#input()
				continue
				
			if e1 == e2:
				continue
			
			tagged_sentence = sentence.replace(e1, '<e1>' + e1 + '</e1>', 1)
			tagged_sentence = tagged_sentence.replace(e2, '<e2>' + e2 + '</e2>', 1)

			text.append(num + '\t"' + tagged_sentence + '"')
			text.append(relation_type + '(e1,e2)')
			text.append('Comment: ' + comment)
			text.append('\n')
			
			print(text)
			#input()
			
print(int(len(text)/4))

print(text[32*4:])
print('==========================================')
print(text[int(len(text)*0.8):])

#######################################################


#from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import f1_score, precision_recall_fscore_support
import numpy as np

#y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
#y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]

y_true = [['1', '2'], ['1', '1'], ['1', '1'], ['0', '0'], ['1', '1'], ['0', '0'], ['2', '1']]
y_pred = [['1', '1'], ['1', '1'], ['1', '1'], ['1', '1'], ['1', '1'], ['1', '1'], ['1', '1']]

y_true = np.concatenate(y_true)
#y_true = list(y_true)
y_pred = np.concatenate(y_pred)
#y_pred = list(y_pred)


print(f1_score(y_true, y_pred, average='weighted'))
#print(accuracy_score(y_true, y_pred))
#print(classification_report(y_true, y_pred))


precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
print(precision)
print(recall)
print(f1)

sys.exit()


asdf = [
		((['Crystal', 'structure', 'of', 'tubulin', 'folding', 'cofactor', 'A', 'from', 'Arabidopsis', 'thaliana', 'and', 'its', 'beta', 'tubulin', 'binding', 'characterization', '.'], (3, 7), (12, 14)), 'tubulin folding cofactor A', 'beta tubulin'),
		
		#((['Crystal', 'structure', 'of', 'tubulin', 'folding', 'cofactor', 'A', 'from', 'Arabidopsis', 'thaliana', 'and', 'its', 'beta', 'tubulin', 'binding', 'characterization', '.'], (12, 14), (3, 7)), 'beta tubulin', 'tubulin folding cofactor A'),
		#((['Crystal', 'structure', 'of', 'tubulin', 'Arabidopsis', 'thaliana'], (0, 1), (3, 4)), 'Crystal', 'tubulin'),
		#((['Crystal', 'structure', 'of', 'tubulin', 'folding', 'cofactor', 'A', 'from', 'Arabidopsis', 'thaliana', 'and', 'its', 'beta', '-', 'tubulin', 'binding', 'characterization', '.'], (12, 15), (3, 7)), 'beta - tubulin', 'tubulin folding cofactor A'),
		#((['Crystal', 'structure', 'of', 'apple', 'folding', 'banana', 'A', 'from', 'pineapple', 'orange', 'and', 'its', 'beta', '-', 'apple', 'binding', 'characterization', '.'], (12, 15), (3, 7)), 'beta-apple', 'apple folding banana A'),
		#((['Crystal', 'structure', 'of', 'apple', 'folding', 'banana', 'A', 'from', 'pineapple', 'orange', 'and', 'its', 'beta', '-', 'apple', 'binding', 'characterization', '.'], (12, 15), (3, 7)), 'beta-apple', 'apple folding banana A'),
		((['It', "'s", 'official', 'U.S.', 'President', 'Barack', 'Obama', 'wants', 'lawmakers', 'to', 'weigh', 'in', 'on', 'whether', 'to', 'use', 'military', 'force', 'in', 'Syria', '.'], (3, 4), (5, 7)), 'U.S.', 'Barack Obama')
	   ]


p1 = pickle.load(open('bioD.pkl', 'rb'))
p2 = pickle.load(open('cnnD.pkl', 'rb'))

#pickle.dump( p1 + p2, open( "D.pkl", "wb" ) )

pickle.dump(asdf + p2, open( "D.pkl", "wb" ) )

'''
print(len(p1))
print(len(p2))

#for i in p1:
#	print(i)

for i in p2:
	for j in i[0][0]:
		if '_' in j:
			print(j)
			print('i[0][0]:', i[0][0])
			print(i)
			input()
	
	#if '-' in i[1] or '-' in i[2]:
	#	print(i)
	#	input()
'''