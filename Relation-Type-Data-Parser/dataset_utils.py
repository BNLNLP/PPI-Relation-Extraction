import sys
from lxml import etree
import csv

import spacy
nlp = spacy.load("en_core_web_sm", disable=["tagger", "ner"])


def parse_bioinfer(xml_file):
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
			"""
			#if entity_type in ["Gene/protein/RNA", "Gene", "Individual_protein", "Protein_family_or_group", "Protein_complex", "Fusion_protein", "Domain_or_region_of_DNA", "DNA_family_or_group"]:
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
									   'offset': str(entity_start_offset) + '-' + str(entity_end_offset),
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

					relations[rel_id] = {'sentence': sentence_txt,
										 'relation_type': rel_type,
										 'rel_nodes': rel_nodes}
					rel_no += 1
		
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


def get_sent(rel_nodes, sents):
	"""
	TODO: handle the case where entities are split in different sentences. 
		  e.g., Greg E.coli - doc: 8576063
				sent 1: Expression of the Escherichia coli
				sent 2: torCAD operon, which encodes the trimethylamine N-oxide reductase system, is regulated by the presence of trimethylamine N-oxide through the action of the TorR response regulator.
				('Agent', 'the Escherichia coli torCAD operon', '129', '34', 'Operon', 'T5'), ('Theme', 'the trimethylamine N-oxide reductase system', '179', '43', 'Enzyme', 'T6')

	"""
	num_of_nodes = len(rel_nodes)

	for idx, sent in enumerate(sents):
		num_of_nodes_in_sent = 0

		checked_nodes = [] # this is used to check preceding and succeeding sentences.
		for node in rel_nodes:
			# start_char: start index of the sentence in document, end_char: end index of the sentence in document
			if sent.start_char <= int(node[2]) and (int(node[2]) + int(node[3])) <= sent.end_char:
				num_of_nodes_in_sent += 1
				checked_nodes.append(node[4])

		if num_of_nodes == num_of_nodes_in_sent:
			return sent.text
		
		# Some sentences are not correctly splitted by spaCy. So, check preceding and succeeding sentences if they have the remaining node. 
		if num_of_nodes_in_sent == 1:
			# check the preceding sentence.
			if idx > 0:
				sent_start_idx = sents[idx-1].start_char
				sent_end_idx = sent.end_char
				
				for node in rel_nodes:
					if node[4] in checked_nodes:
						continue
					if sent_start_idx <= int(node[2]) and (int(node[2]) + int(node[3])) <= sent_end_idx:
						num_of_nodes_in_sent += 1
						checked_nodes.append(node[4])
						
				if num_of_nodes == num_of_nodes_in_sent:
					return sents[idx-1].text + ' ' + sent.text
				
			# check the succeeding sentence.
			if idx < len(sents)-1:
				sent_start_idx = sent.start_char
				sent_end_idx = sents[idx+1].end_char
				
				for node in rel_nodes:
					if node[4] in checked_nodes:
						continue
					if sent_start_idx <= int(node[2]) and (int(node[2]) + int(node[3])) <= sent_end_idx:
						num_of_nodes_in_sent += 1
						checked_nodes.append(node[4])
						
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


def parse_grec(xml_file):
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
						rel_nodes.append((role, ref_text, ref_offset, ref_length, ref_type, refid))
						
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

						rel_nodes = [x for x in rel_nodes if x[0].lower() in ['agent', 'theme']]

						relations[doc_rel_id] = {'text': text,
												 'sentence': sentence,
												 'event_type': v['event_type'],
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
	'''
	
	species_type = xml_file.rsplit('/', 1)[1].split('_', 1)[1].split('_', 1)[0]

	with open('greg_' + species_type + '.csv', mode='w') as csv_file:
		fieldnames = ['No', 'Entities', 'Relation', 'Sentence']
		writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

		writer.writeheader()
		
		duplicate_checker = set()
		num = 1
		for k, v in relations.items():
			if v['sentence'] is None:
				continue
			
			tmp = ' | '.join([x[1] + ' (' + x[0] + ')' for x in sorted(v['rel_nodes'])]) + v['event_type'] + v['sentence'].strip()
			
			if tmp not in duplicate_checker:
				writer.writerow({'No': num, 'Entities': ' | '.join([x[1] + ' (' + x[0] + ')' for x in v['rel_nodes']]), 'Relation': v['event_type'], 'Sentence': v['sentence'].strip()})
				duplicate_checker.add(tmp)
				num += 1


def parse_genereg(xml_file):
	
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
					rel_nodes.append((role, ref_text, ref_offset, ref_length, ref_type, refid))

				sentence = get_sent(rel_nodes, sentences) # sentence of the relation
				
				doc_rel_id = document_id + '_' + id
				relations[doc_rel_id] = {'text': text,
										 'sentence': sentence,
										 'relation_type': relation_type,
										 'rel_nodes': rel_nodes}
	
	# debug
	all_types = set()
	all_rel_types = set()
	all_roles = set()
	num_of_no_sent = 0
	
	for k, v in relations.items():
		#print(k, v)
		all_types.add(v['type'])
		all_rel_types.add(v['relation_type'])
		for r in v['rel_nodes']:
			all_roles.add(r[0])
		if v['sentence'] is None:
			num_of_no_sent += 1
	
	print(all_types)
	print(all_rel_types)
	print(all_roles)
	print(len(relations))
	print(num_of_no_sent)
	
	
	with open('genereg.csv', mode='w') as csv_file:
		fieldnames = ['No', 'Entities', 'Relation', 'Sentence']
		writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

		writer.writeheader()
		
		duplicate_checker = set()
		num = 1
		for k, v in relations.items():
			if v['sentence'] is None:
				continue
			
			tmp = ' | '.join([x[1] + ' (' + x[0] + ')' for x in sorted(v['rel_nodes'])]) + v['relation_type'] + v['sentence'].strip()
			
			if tmp not in duplicate_checker:
				writer.writerow({'No': num, 'Entities': ' | '.join([x[1] + ' (' + x[0] + ')' for x in v['rel_nodes']]), 'Relation': v['relation_type'], 'Sentence': v['sentence'].strip()})
				duplicate_checker.add(tmp)
				num += 1


def main():
	#bioinfer_corpus = 'datasets/BioInfer/BioInfer_corpus_1.1.1.xml'
	bioinfer_corpus = 'datasets/BioInfer/BioInfer_corpus_1.2.0b.binarised.xml'
	parse_bioinfer(bioinfer_corpus)
	
	#grec_corpus = 'datasets/GREC/grec_ecoli_bioc.xml'
	grec_corpus = 'datasets/GREC/grec_human_bioc.xml'
	#parse_grec(grec_corpus)
	
	genereg_corpus = 'datasets/GeneReg/genereg_bioc.xml'
	#parse_genereg(genereg_corpus)
	
	
if __name__ == "__main__":
	main()

