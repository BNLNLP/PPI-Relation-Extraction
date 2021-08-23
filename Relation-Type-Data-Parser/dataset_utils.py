import os
import sys
from lxml import etree
import csv
import json

import spacy
nlp = spacy.load("en_core_web_sm", disable=["tagger", "ner"])


def add_em_to_sentence(sentence, e1, e2):
	"""
	Add Entity Start Markers to sentence.
	
	referred to the function 'get_samples_from_ppi_benchmark()' from 'PPI-Relation-Extraction\BERT-Relation-Extraction\src\tasks\preprocessing_funcs.py'
	"""
	
	'''
	pair_id = pair_elem.get('id')
	pair_e1 = pair_elem.get('e1')
	pair_e2 = pair_elem.get('e2')
	pair_interaction = pair_elem.get('interaction')
	
	# TODO: handle entities that consist of separate words in a sentence.
	if pair_e1 not in entities or pair_e2 not in entities:
		continue
	'''
	
	e1_start_idx = e1[1]
	e2_start_idx = e2[1]
	e1_end_idx = e1[2]
	e2_end_idx = e2[2]
	e1_text = e1[0]
	e2_text = e2[0]
				
	# TODO: handle self PPIs and cases where entities are overlapped.
	''' E.g., AImed.xml: 
			<sentence id="AIMed.d198.s1687" seqId="s1687" text="Mutagenesis of the erythropoietin receptor (EPOR) permits analysis of the contribution that individual amino acid residues make to erythropoietin (EPO) binding.">
				<entity charOffset="19-32" id="AIMed.d198.s1687.e0" seqId="e5485" text="erythropoietin" type="protein" />
				<entity charOffset="19-41" id="AIMed.d198.s1687.e1" seqId="e5487" text="erythropoietin receptor" type="protein" />
	'''
	''' debugging
	if pair_e1 == pair_e2:
		print('pair_id:', pair_id)
		print('pair_e1 == pair_e2:', pair_e1, pair_e2)
		input('enter..')
		
	if e1_start_idx == e2_start_idx:
		print('pair_id:', pair_id)
		print('e1_start_idx == e2_start_idx:', e1_start_idx, e2_start_idx)
		input('enter..')
	
	if e1_start_idx > e2_start_idx and e1_start_idx < e2_end_idx:
		print('pair_id:', pair_id)
		print('e1_start_idx > e2_start_idx and e1_start_idx < e2_end_idx:', e1_start_idx, e2_start_idx, e2_end_idx)
		input('enter..')
	
	if e2_start_idx > e1_start_idx and e2_start_idx < e1_end_idx:
		print('pair_id:', pair_id)
		print('e2_start_idx > e1_start_idx and e2_start_idx < e1_end_idx:', e2_start_idx, e1_start_idx, e1_end_idx)
		input('enter..')
	'''
	if e1_text == e2_text \
	   or e1_start_idx == e2_start_idx \
	   or (e1_start_idx > e2_start_idx and e1_start_idx < e2_end_idx) \
	   or (e2_start_idx > e1_start_idx and e2_start_idx < e1_end_idx):
		print('>> self interaction or entities are overlapped!!')
		print(e1_text, '/', e2_text)
		print(sentence[0])
		return None
		
	tagged_sent = sentence[0]

	if e1_start_idx < e2_start_idx: # replace first the one located in the rear.
		tagged_sent = tagged_sent[:e2_start_idx] + '<e2>' + e2_text + '</e2>' + tagged_sent[e2_end_idx:]
		tagged_sent = tagged_sent[:e1_start_idx] + '<e1>' + e1_text + '</e1>' + tagged_sent[e1_end_idx:]
	else:
		tagged_sent = tagged_sent[:e1_start_idx] + '<e1>' + e1_text + '</e1>' + tagged_sent[e1_end_idx:]
		tagged_sent = tagged_sent[:e2_start_idx] + '<e2>' + e2_text + '</e2>' + tagged_sent[e2_end_idx:]
		
	print(e1_text, '/', e2_text)
	print(sentence[0])
	print(tagged_sent)
	
	#input('enter..')
	
	return tagged_sent
	
				

def parse_bioinfer(xml_file, add_entity_marker=True, write_to_file=False):
	"""
	TODO: handle entities composed of subtokens from different positions (not consecutive). 
		  e.g., the entity "alpha 5 integrins" in the sentence "Abundance of actin, talin, alpha 5 and beta 1 integrins, ..."
	"""
	xml_parser = etree.XMLParser(ns_clean=True)
	tree = etree.parse(xml_file, xml_parser)
	root = tree.getroot()
	
	relations = {}
	sentences = root.find('.//sentences')
	for sentence in sentences.findall('.//sentence'):
		sentence_id = sentence.get('id')
		sentence_txt = sentence.get('origText')
		
		print(sentence_txt)
		
		subtokens = {}
		for token in sentence.findall('.//token'):
			token_id = token.get('id') # e.g., "t.2.10"
			offset = int(token.get('charOffset'))
			for subtoken in token.findall('.//subtoken'):
				subtoken_id = subtoken.get('id') # e.g., "st.2.10.0", "st.2.10.1"
				text = subtoken.get('text')
				subtokens[subtoken_id] = {'token_id': token_id, 
										  'offset': offset, 
										  'text': text}
				offset += len(text)
				
		entities = {}
		for entity in sentence.findall('.//entity'):
			entity_id = entity.get('id')
			entity_type = entity.get('type')
			
			"""
			This was needed to parse 'BioInfer_corpus_1.1.1.xml' file, and it's not needed anymore to check types in 'BioInfer_corpus_1.2.0b.binarised.xml' 
			since the entities are already verified as gene/protein by the authors. 07-23-2021
			
			TODO: check if there are other available types referring to the entity type ontology in the paper.
			- The types of entities used BioInfer benchmark file (BioInferCLAnalysis_split_SMBM_version.xml.viewer) have been referred.
			
			#if entity_type in ["Gene/protein/RNA", "Gene", "Individual_protein", "Protein_family_or_group", "Protein_complex", "Fusion_protein", "Domain_or_region_of_DNA", "DNA_family_or_group"]:
			"""
			
			largest_token_id_seq = -1
			is_offset_split = False # to check if nested subtokens come from separate tokens (not consecutive). e.g., <nestedsubtoken id="st.2.7.0" />, <nestedsubtoken id="st.2.11.0" />        
			entity_text = ''
			entity_start_offset, entity_end_offset = -1, -1
			for nestedsubtoken in entity.findall('.//nestedsubtoken'):
				subtoken_id = nestedsubtoken.get('id') # e.g., "st.2.10.0", "st.2.10.1"
				offset = subtokens[subtoken_id]['offset']
				text = subtokens[subtoken_id]['text']
				text_len = len(text)

				token_id_seq = int(subtokens[subtoken_id]['token_id'].rsplit('.', 1)[1]) # e.g., 10 in the token id "t.2.10"
				
				if largest_token_id_seq == -1 or token_id_seq == largest_token_id_seq or token_id_seq == largest_token_id_seq + 1:
					largest_token_id_seq = token_id_seq
					if entity_start_offset == -1:
						entity_start_offset, entity_end_offset = offset, offset + text_len
					elif entity_start_offset > offset:
						entity_start_offset = offset
					elif entity_end_offset < offset + text_len:
						entity_end_offset = offset + text_len
				else:
					is_offset_split = True
					break
			
			if is_offset_split is False:
				entities[entity_id] = {'text': sentence_txt[entity_start_offset:entity_end_offset],
									   #'offset': str(entity_start_offset) + '-' + str(entity_end_offset),
									   'start_offset': entity_start_offset, 
									   'end_offset': entity_end_offset,
									   'type': entity_type}
		
		# debug
		'''
		for k, v in entities.items():
			print(k, v)
		print('------------------------------------------------------')
		'''
		
		rel_no = 1
		formulas = sentence.find('.//formulas')
		for formula in formulas.findall('.//formula'):
			for relnode in formula.iterfind(".//relnode"):
				rel_entity_id = relnode.get('entity')
				rel_type = relnode.get('predicate')
				
				# debug
				'''
				if rel_entity_id in entities:
					rel_entity = entities[rel_entity_id]
					print(relnode.get('entity'), rel_entity, rel_type)
				else:
					print(relnode.get('entity'), 'No relation entity in entities |', rel_type)
				'''
				
				rel_nodes = []
				for entity in relnode.getchildren():
					if entity.tag == 'entitynode':
						entity_id = entity.get('entity')
						if entity_id in entities:
							rel_nodes.append(entities[entity_id])
						#else:
						#	print(entity_id, 'No entity in entities')
						#	input('enter..')
				
				if len(rel_nodes) == 2: # binary relation
					rel_id = sentence_id + '_' + str(rel_no)
					
					if rel_id in relations:
						sys.exit('duplicate error!!')
					
					rel_sent = sentence_txt
					if add_entity_marker:
						rel_sent = add_em_to_sentence([sentence_txt, 0, len(sentence_txt)],
													  [rel_nodes[0]['text'], rel_nodes[0]['start_offset'], rel_nodes[0]['end_offset']],
													  [rel_nodes[1]['text'], rel_nodes[1]['start_offset'], rel_nodes[1]['end_offset']])
					
						duplicate = False
						if rel_sent in [x['sentence'] for x in relations.values()]:
							duplicate = True
							
							print(rel_sent)
							print(rel_type)
							print([(x['sentence'], x['relation_type']) for x in relations.values()])
							#input('enter..')
							
					if rel_sent is not None and duplicate == False:
						relations[rel_id] = {'sentence': rel_sent,
											 'relation_type': rel_type,
											 'rel_nodes': rel_nodes}
						rel_no += 1
	
	# debug
	'''
	all_rel_types = set()
	
	for k, v in relations.items():
		all_rel_types.add(v['relation_type'])
	print(all_rel_types)
	
	all_node_types = {}
	for val in relations.values():
		for node in val['rel_nodes']:
			print(node['type'], ' | ', node['text'])
			
			if node['type'] in all_node_types:
				all_node_types[node['type']] += 1
			else:
				all_node_types[node['type']] = 1
	print(all_node_types)
	
	input('enter..')
	'''
	
	if write_to_file:
		with open('bioinfer.csv', mode='w') as csv_file:
			fieldnames = ['No', 'Entities', 'Relation', 'Sentence']
			writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

			writer.writeheader()
			
			duplicate_checker = set()
			num = 1
			for k, v in relations.items():
				if v['sentence'] is None:
					continue
				
				tmp = ' | '.join([x['text'] + ' (' + x['type'] + ')' for x in v['rel_nodes']]) + v['relation_type'] + v['sentence'].strip()
				
				if tmp not in duplicate_checker:
					writer.writerow({'No': num, 'Entities': ' | '.join([x['text'] + ' (' + x['type'] + ')' for x in v['rel_nodes']]), 'Relation': v['relation_type'], 'Sentence': v['sentence'].strip()})
					duplicate_checker.add(tmp)
					num += 1
					
	return relations
	

def get_sent(rel_nodes, sents):
	"""
	TODO: handle the case where entities are split in different sentences. 
		  e.g., Greg E.coli - doc: 8576063
				sent 1: Expression of the Escherichia coli
				sent 2: torCAD operon, which encodes the trimethylamine N-oxide reductase system, is regulated by the presence of trimethylamine N-oxide through the action of the TorR response regulator.
				
				{'refid': 'T5', 'role': 'Agent', 'ref_text': 'the Escherichia coli torCAD operon', 'ref_offset': '129', 'ref_length': '34', 'ref_type': 'Operon'})
				{'refid': 'T6', 'role': 'Theme', 'ref_text': 'the trimethylamine N-oxide reductase system', 'ref_offset': '179', 'ref_length': '43', 'ref_type': 'Enzyme'})
	"""
	num_of_nodes = len(rel_nodes)

	for idx, sent in enumerate(sents):
		num_of_nodes_in_sent = 0

		checked_nodes = [] # this is used to check preceding and succeeding sentences.
		for node in rel_nodes:
			# start_char: start index of the sentence in document, end_char: end index of the sentence in document
			if sent.start_char <= int(node['ref_offset']) and (int(node['ref_offset']) + int(node['ref_length'])) <= sent.end_char:
				num_of_nodes_in_sent += 1
				checked_nodes.append(node['refid'])

		if num_of_nodes == num_of_nodes_in_sent:
			return sent.text
		
		# Some sentences are not correctly splitted by spaCy. So, check preceding and succeeding sentences if they have the remaining node. 
		if num_of_nodes_in_sent == 1:
			# check the preceding sentence.
			if idx > 0:
				sent_start_idx = sents[idx-1].start_char
				sent_end_idx = sent.end_char
				
				for node in rel_nodes:
					if node['refid'] in checked_nodes:
						continue
					if sent_start_idx <= int(node['ref_offset']) and (int(node['ref_offset']) + int(node['ref_length'])) <= sent_end_idx:
						num_of_nodes_in_sent += 1
						checked_nodes.append(node['refid'])
						
				if num_of_nodes == num_of_nodes_in_sent:
					return sents[idx-1].text + ' ' + sent.text
				
			# check the succeeding sentence.
			if idx < len(sents)-1:
				sent_start_idx = sent.start_char
				sent_end_idx = sents[idx+1].end_char
				
				for node in rel_nodes:
					if node['refid'] in checked_nodes:
						continue
					if sent_start_idx <= int(node['ref_offset']) and (int(node['ref_offset']) + int(node['ref_length'])) <= sent_end_idx:
						num_of_nodes_in_sent += 1
						checked_nodes.append(node['refid'])
						
				if num_of_nodes == num_of_nodes_in_sent:
					return sent.text + ' ' + sents[idx+1].text

	'''
	for s in sents:
		print(s)
		for t in s:
			print(t)
	'''
	#print(rel_nodes)
	#input('enter..')

	return None


def parse_grec(xml_file, write_to_file=False):
	"""
	TODO: handle the event entities.
		  e.g., Greg E.coli - doc: 10463182
				<relation id="E15">
					<infon key="event type">GRE</infon>
					<infon key="file">a2</infon>
					<infon key="type">Event</infon>
					<node refid="T44" role="Trigger"/>
					<node refid="E14" role="Agent"/>
					<node refid="E16" role="Theme"/>
				</relation>
	"""
	
	xml_parser = etree.XMLParser(ns_clean=True)
	tree = etree.parse(xml_file, xml_parser)
	root = tree.getroot()
	
	relations = {}
	for document in root.findall('.//document'):
		document_id = document.findtext('.//id')
		
		print('document_id:', document_id)
		
		for passage in document.findall('.//passage'):
			text = passage.findtext('.//text')
			sentences = list(nlp(text).sents) # spaCy uses a generator which is only used once.
			
			annotation_dict = {}
			for annotation in passage.findall('.//annotation'):
				id = annotation.get('id')
				annotation_dict[id] = {'file': annotation.findtext('.//infon[@key="file"]'),
									   'type': annotation.findtext('.//infon[@key="type"]'),
									   'offset': annotation.find('.//location').get('offset'),
									   'length': annotation.find('.//location').get('length'),
									   'text': annotation.findtext('.//text')}
			
			rel_info = {}
			for relation in passage.findall('.//relation'):
				id = relation.get('id')
				file = relation.findtext('.//infon[@key="file"]')
				type_ = relation.findtext('.//infon[@key="type"]')
				event_type = relation.findtext('.//infon[@key="event type"]')
				
				if type_ != 'Event': # type must be 'Event'
					sys.exit("wrong type: " + type_)
					
				rel_nodes = []
				#all_roles = []
				for node in relation.findall('.//node'):
					role = node.get('role')
					#all_roles.append(role.lower())
					refid = node.get('refid')
					#ref_text = annotation_dict[refid].get('text')
					#ref_offset = annotation_dict[refid].get('offset')
					#ref_length = annotation_dict[refid].get('length')
					#rel_nodes.append((role, ref_text, ref_offset, ref_length))
					rel_nodes.append((role, refid))
				
				rel_info[id] = {'event_type': event_type, 'rel_nodes': rel_nodes}
			
			def find_concept(role, refid, rel_info):
				for node in rel_info[refid]['rel_nodes']:
					n_role = node[0]
					n_refid = node[1]
					if role == n_role and n_refid.startswith('T'):
						return n_refid
					elif role == n_role and n_refid.startswith('E'):
						find_concept(n_role, n_refid, rel_info)
					else:
						print(role, refid)
						input('enter..')
			
			for k, v in rel_info.items():
				rel_nodes = []
				all_roles = []
				for node in v['rel_nodes']:
					role = node[0]
					refid = node[1]
					
					if refid.startswith('T'):
						all_roles.append(role.lower())
						
						ref_text = annotation_dict[refid].get('text')
						ref_offset = annotation_dict[refid].get('offset')
						ref_length = annotation_dict[refid].get('length')
						ref_type = annotation_dict[refid].get('type')
						#rel_nodes.append((role, ref_text, ref_offset, ref_length, ref_type, refid))
						rel_nodes.append({'refid': refid, 
										  'role': role, 
										  'ref_text': ref_text, 
										  'ref_offset': ref_offset, 
										  'ref_length': ref_length, 
										  'ref_type': ref_type})
						
					'''
					elif refid.startswith('E'): # if it's an event,
						refid = find_concept(role, refid, rel_info)
						
						if refid is None:
							print(role, refid)
						
						rel_nodes.append(ret_val)
					'''

				if 'agent' in all_roles and 'theme' in all_roles: # only consider relations between agent and theme.	
					sentence = get_sent(rel_nodes, sentences) # sentence of the relation
					
					if sentence is not None:
						doc_rel_id = document_id + '_' + id

						rel_nodes = [x for x in rel_nodes if x['role'].lower() in ['agent', 'theme']]

						relations[doc_rel_id] = {'text': text,
												 'sentence': sentence,
												 'relation_type': v['event_type'],
												 'rel_nodes': rel_nodes}
					# debug
					'''
					else:
						print(text)
						for s in sentences:
							print(s)
						print(rel_nodes)
						#input('enter..')
					'''	
	# debug
	'''
	print(len(relations))
	input('enter..')
	
	for rel in relations:
		print(rel)
		input('enter..')
	
	all_rel_types = set()
	
	for k, v in relations.items():
		all_rel_types.add(v['event_type'])
	print(all_rel_types)
	
	all_node_types = {}
	for val in relations.values():
		for node in val['rel_nodes']:
			print(node['ref_type'], ' | ', node['ref_text'])
			
			if node['ref_type'] in all_node_types:
				all_node_types[node['ref_type']] += 1
			else:
				all_node_types[node['ref_type']] = 1
	print(all_node_types)
	
	input('enter...')
	'''	

	if write_to_file:
		species_type = xml_file.rsplit('/', 1)[1].split('_', 1)[1].split('_', 1)[0]

		with open('grec_' + species_type + '.csv', mode='w') as csv_file:
			fieldnames = ['No', 'Entities', 'Relation', 'Sentence']
			writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

			writer.writeheader()
			
			duplicate_checker = set()
			num = 1
			for k, v in relations.items():
				if v['sentence'] is None:
					continue
				
				node_txt_role = [(x['role'], x['ref_text']) for x in v['rel_nodes']]
				unique_rel_identifer = ' | '.join([x[1] + ' (' + x[0] + ')' for x in sorted(node_txt_role)]) + v['relation_type'] + v['sentence'].strip()

				if unique_rel_identifer not in duplicate_checker:
					writer.writerow({'No': num, 'Entities': ' | '.join([x['ref_text'] + ' (' + x['role'] + ')' for x in v['rel_nodes']]), 'Relation': v['relation_type'], 'Sentence': v['sentence'].strip()})
					duplicate_checker.add(unique_rel_identifer)
					num += 1
	
	return relations


def parse_genereg(xml_file, write_to_file=False):
	
	xml_parser = etree.XMLParser(ns_clean=True)
	tree = etree.parse(xml_file, xml_parser)
	root = tree.getroot()
	
	relations = {}
	for document in root.findall('.//document'):
		document_id = document.findtext('.//id')

		for passage in document.findall('.//passage'):
			text = passage.findtext('.//text')
			sentences = list(nlp(text).sents) # spaCy uses a generator which is only used once.
	
			annotation_dict = {}
			for annotation in passage.findall('.//annotation'):
				id = annotation.get('id')
				annotation_dict[id] = {'file': annotation.findtext('.//infon[@key="file"]'),
									   'type': annotation.findtext('.//infon[@key="type"]'),
									   'offset': annotation.find('.//location').get('offset'),
									   'length': annotation.find('.//location').get('length'),
									   'text': annotation.findtext('.//text')}
			
			for relation in passage.findall('.//relation'):
				id = relation.get('id')
				file = relation.findtext('.//infon[@key="file"]')
				type_ = relation.findtext('.//infon[@key="type"]')
				relation_type = relation.findtext('.//infon[@key="relation type"]')
				
				if type_ != 'Relation': # type must be 'Relation'
					sys.exit("wrong type: " + type_)
				
				rel_nodes = []
				for node in relation.findall('.//node'):
					role = node.get('role')
					refid = node.get('refid')
					ref_text = annotation_dict[refid].get('text')
					ref_offset = annotation_dict[refid].get('offset')
					ref_length = annotation_dict[refid].get('length')
					ref_type = annotation_dict[refid].get('type')

					if ref_type != 'Gene': # skip 'Entity'
						continue
								
					rel_nodes.append({'refid': refid, 
									  'role': role, 
									  'ref_text': ref_text, 
									  'ref_offset': ref_offset, 
									  'ref_length': ref_length, 
									  'ref_type': ref_type})
					
				sentence = get_sent(rel_nodes, sentences) # sentence of the relation
				
				doc_rel_id = document_id + '_' + id
				relations[doc_rel_id] = {'text': text,
										 'sentence': sentence,
										 'relation_type': relation_type,
										 'rel_nodes': rel_nodes}
	
	# debug
	'''
	#all_types = set()
	all_rel_types = set()
	all_roles = set()
	num_of_no_sent = 0
	
	for k, v in relations.items():
		#print(k, v)
		#all_types.add(v['type'])
		all_rel_types.add(v['relation_type'])
		for r in v['rel_nodes']:
			all_roles.add(r['role'])
		if v['sentence'] is None:
			num_of_no_sent += 1
	
	#print(all_types)
	print(all_rel_types)
	print(all_roles)
	print(len(relations))
	print(num_of_no_sent)
	
	all_node_types = {}
	for val in relations.values():
		for node in val['rel_nodes']:
			print(node['ref_type'], ' | ', node['ref_text'])
			
			if node['ref_type'] in all_node_types:
				all_node_types[node['ref_type']] += 1
			else:
				all_node_types[node['ref_type']] = 1
	print(all_node_types)
	
	input('enter...')
	'''

	if write_to_file:
		with open('genereg.csv', mode='w') as csv_file:
			fieldnames = ['No', 'Entities', 'Relation', 'Sentence']
			writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

			writer.writeheader()
			
			duplicate_checker = set()
			num = 1
			for k, v in relations.items():
				if v['sentence'] is None:
					continue

				node_txt_role = [(x['role'], x['ref_text']) for x in v['rel_nodes']]
				unique_rel_identifer = ' | '.join([x[1] + ' (' + x[0] + ')' for x in sorted(node_txt_role)]) + v['relation_type'] + v['sentence'].strip()

				if unique_rel_identifer not in duplicate_checker:
					writer.writerow({'No': num, 'Entities': ' | '.join([x['ref_text'] + ' (' + x['role'] + ')' for x in v['rel_nodes']]), 'Relation': v['relation_type'], 'Sentence': v['sentence'].strip()})
					duplicate_checker.add(unique_rel_identifer)
					num += 1
	
	return relations
	

def parse_genia(dir):
	"""
	
	"""
	relations = {}
	total_rel_cnt = 0 # debug
	for filename in os.listdir(dir):
		if filename.endswith('a2'): # files contain standoff annotations for other entities and events.
			
			print('filename:', filename)
			
			events = {}
			terms = {}
			with open(os.path.join(dir, filename)) as file:
				for line in file.readlines():
					if line.startswith('E'):
						tokens = line.split()
						id = tokens[0]
						event = tokens[1].split(':')
						arguments = [(x.split(':')[1], x.split(':')[0]) for x in tokens[2:]] # argument_term_id, argument_type
						
						events[id] = {'event': event, 'arguments': arguments}
							
					elif line.startswith('T'):
						tokens = [x.strip() for x in line.split(None, 4)]
						id = tokens[0]
						term_type = tokens[1]
						start_offset = tokens[2]
						end_offset = tokens[3]
						text = tokens[4]
						terms[id] = {'type': term_type, 'start_offset': start_offset, 'end_offset': end_offset, 'text': text}
			
			if len(events) > 0:
				protein_anno_file = filename.replace('.a2', '.a1') # files contain standoff annotations for proteins.
				with open(os.path.join(dir, protein_anno_file)) as file:
					for line in file.readlines():
						if line.startswith('T'):
							tokens = [x.strip() for x in line.split(None, 4)]
							id = tokens[0]
							term_type = tokens[1]
							start_offset = tokens[2]
							end_offset = tokens[3]
							text = tokens[4]
							terms[id] = {'type': term_type, 'start_offset': start_offset, 'end_offset': end_offset, 'text': text}
				
				json_file = filename.replace('.a2', '.json') # files contain both text and annotations in JSON.
				with open(os.path.join(dir, json_file)) as file:
					data = json.load(file)
					text = data['text']

					sentences = list(nlp(text).sents) # spaCy uses a generator which is only used once.

					for k, v in events.items():
						rel_nodes = []
						
						if len(v['arguments']) < 2:
							continue
						
						total_rel_cnt += 1 # debug
						has_event_arg = False # debug
						
						for argument in v['arguments']: # argument -> (argument_term_id, argument_type) e.g., (E27, Cause) 

							def get_arg_info(argument, has_event_arg):
								refid = argument[0] # e.g., E27
								role = argument[1]  # e.g., Cause
								
								if refid.startswith('E'):
									
									print('>> this event has an event argument:', k, v, refid)
									
									if len(events[refid]['arguments']) == 1:
										return get_arg_info(events[refid]['arguments'][0], True)
									else:
										'''
										TODO: handle the event themes that have more than 2 arguments. 
										
										E.g., 
										"We also found that the recruitment of HOIP to CD40 was TRAF2-dependent and that overexpression of a truncated HOIP mutant partially inhibited CD40-mediated NF-kappaB activation."
										(Original annotation)
										TRAF2 (Cause) -> dependent (Regulation) -> (recruitment (Binding) -> HOIP (Theme), CD40 (Theme)) (Theme)

										"For TNF and IL-2, the requirement for Runx3 subsides by day 6 (Fig. 3 C), possibly because of compensation by Runx1, which is derepressed in Runx3-/- cells (Fig. 4 A)."
										(Original annotation)
										Runx1 (Cause) -> subsides (Negative_regulation) -> (Runx3 (Cause) -> requirement (Positive_regulation) -> TNF (Theme)) (Theme)
										'''
										return None, None, None, False
								else:
									return refid, role, terms[refid], has_event_arg
							
							refid, role, term, has_event_arg = get_arg_info(argument, has_event_arg)
							
							if refid is None:
								continue
							
							if term['type'] != 'Protein': # skip 'Entity'
								continue
							
							rel_nodes.append({'refid': refid, 
											  'role': role, 
											  'ref_text': term['text'], 
											  'ref_offset': term['start_offset'], 
											  'ref_length': int(term['end_offset']) - int(term['start_offset']), 
											  'ref_type': term['type']})

						if len(rel_nodes) < 2:
							continue

						sentence = get_sent(rel_nodes, sentences) # sentence of the relation
						
						rel_id = filename + '_' + k
						relations[rel_id] = {'text': text,
											 'sentence': sentence,
											 'relation_type': v['event'][0], 
											 'rel_nodes': rel_nodes,
											 'has_event_arg': has_event_arg} # debug
					
					# debug
					'''
					for k, v in relations.items():
						if v['has_event_arg']:
							print('>> relation id:', k, '\n', v)
							input('enter..')
					'''
	# debug
	'''
	print(len({k: v for k, v in relations.items() if v['has_event_arg']}))
	print(len(relations))
	print(total_rel_cnt)
	
	d = {}
	for x in [v['relation_type'] for v in relations.values()]:
		if x in d:
			d[x] += 1
		else:
			d[x] = 1
	
	for k, v in d.items():
		print(k, v)
		
	all_node_types = {}
	for val in relations.values():
		for node in val['rel_nodes']:
			print(node['ref_type'], ' | ', node['ref_text'])
			
			if node['ref_type'] in all_node_types:
				all_node_types[node['ref_type']] += 1
			else:
				all_node_types[node['ref_type']] = 1
	print(all_node_types)
	
	input('enter..')
	'''

	return relations


def consolidate_relations(relation_list):
	"""
	Merge all relations from datasets: BioInfer, GREC, GeneReg, GENIA
	
	"""
	all_relations = {}
	
	for relation in relation_list:
		if len(all_relations.keys() & relation.keys()) > 0: # check if there's duplicate keys.
			print(all_relations.keys() & relation.keys())
			input('enter...')
		
		all_relations.update(relation)
	
	rel_type = []
	for k, v in all_relations.items():
		rel_type.append(v['relation_type'])
	print(set(rel_type))
	
	

def main():
	add_entity_marker = True
	write_to_file = False

	'''
	relations[rel_id] = {'sentence': sentence_txt,
						 'relation_type': rel_type,
						 'rel_nodes': rel_nodes}
						 
						 entities[entity_id] = {'text': sentence_txt[entity_start_offset:entity_end_offset],
											   'offset': str(entity_start_offset) + '-' + str(entity_end_offset),
											   'type': entity_type}
	'''
	bioinfer_corpus = 'datasets/BioInfer/BioInfer_corpus_1.2.0b.binarised.xml'
	#bioinfer_relations = parse_bioinfer(bioinfer_corpus, add_entity_marker, write_to_file)
	

	'''
	relations[doc_rel_id] = {'text': text,
							 'sentence': sentence,
							 'relation_type': v['event_type'],
							 'rel_nodes': rel_nodes}
							 
							 rel_nodes.append({'refid': refid, 
											  'role': role, 
											  'ref_text': ref_text, 
											  'ref_offset': ref_offset, 
											  'ref_length': ref_length, 
											  'ref_type': ref_type})
	'''
	grec_ecoli_corpus = 'datasets/GREC/grec_ecoli_bioc.xml'
	grec_ecoli_relations = parse_grec(grec_ecoli_corpus, write_to_file)
	
	grec_human_corpus = 'datasets/GREC/grec_human_bioc.xml'
	grec_human_relations = parse_grec(grec_human_corpus, write_to_file)

	'''
	doc_rel_id = document_id + '_' + id
	relations[doc_rel_id] = {'text': text,
							 'sentence': sentence,
							 'relation_type': relation_type,
							 'rel_nodes': rel_nodes}
							 
							 rel_nodes.append({'refid': refid, 
											  'role': role, 
											  'ref_text': ref_text, 
											  'ref_offset': ref_offset, 
											  'ref_length': ref_length, 
											  'ref_type': ref_type})
	'''
	genereg_corpus = 'datasets/GeneReg/genereg_bioc.xml'
	genereg_relations = parse_genereg(genereg_corpus, write_to_file)

	'''
	relations[rel_id] = {'text': text,
						 'sentence': sentence,
						 'relation_type': v['event'], 
						 'rel_nodes': rel_nodes,
						 'has_event_arg': has_event_arg} # debug
						 
						 rel_nodes.append({'refid': refid, 
										  'role': role, 
										  'ref_text': term['text'], 
										  'ref_offset': term['start_offset'], 
										  'ref_length': int(term['end_offset']) - int(term['start_offset']), 
										  'ref_type': term['type']})
	'''
	genia_train_corpus = 'datasets/GENIA/BioNLP-ST-2013_GE_train_data_rev3'
	genia_train_relations = parse_genia(genia_train_corpus)

	genia_devel_corpus = 'datasets/GENIA/BioNLP-ST-2013_GE_devel_data_rev3'
	genia_devel_relations = parse_genia(genia_devel_corpus)

	consolidate_relations([bioinfer_relations, grec_ecoli_relations, grec_human_relations, 
						   genereg_relations, genia_train_relations, genia_devel_relations])
	
	
if __name__ == "__main__":
	main()

