import re
import json
import spacy


def write_to_file(result, out_file):
	with open(out_file, 'w+') as file:
		html = """<html>
					<head>
						<title>Relation Extraction</title>
						<!-- 
						<script type="text/javascript" id="MathJax-script" async
							src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/mml-chtml.js">
						</script>
						-->
						<script type="text/javascript" async
							src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML">
						</script>
						<style>
						table
						{
							table-layout: fixed;
							width: 100%;
						}
						#no {width:3%}
						#data {width:48.5%}
						
						h2.st1 {
						  padding: 15px;
						}
						p.st1 {
						  padding: 15px;
						}
						</style>
					</head>
					<h2 class="st1">BioCreative VI - Track 4: Mining protein interactions and mutations for precision medicine (PM) [2017-03-03] </h2>
					<p class="st1">
					A subset of the relevant articles in Document Triage Task has been manually annotated with relevant interacting protein pairs. 
					Each PubMed article in this set has at least one interacting pair which is listed with the GeneEntrez ID of the two interactors. 
					These protein-protein interactions have been experimentally verified and the analysis of natural occurring or synthetic mutations has identified protein residues crucial for the interaction. 
					Participants in this task will be expected to build automated methods that are capable of receiving a set of PMID documents and return the set of interacting protein pairs (and their corresponding Gene Entrez IDs) mentioned in the text that are affected by a genetic mutation. <br />
					<a href=https://biocreative.bioinformatics.udel.edu/tasks/biocreative-vi/track-4/> Further information </a>
					</p>
					<table border="1">
						<tr><th id="no">No.</th><th id="data">Text w/ genes highlighted</th><th id="data">Relation information</th></tr>"""
		counter = 1
		for item in result:
			html += "<tr>"
			html += "<td align='center'>{}</td>".format(counter)
			html += "<td>{}</td>".format(item['html_text'])
			color_info = ''
			for i in item['color_info']:
				color_info += 'Relation: <span style="background-color:' + i[1] + '"> Gene_ID (' + i[0] + ')</span>' + \
								  	   '-<span style="background-color:' + i[3] + '"> Gene_ID (' + i[2] + ')</span><br>' 
			html += "<td>{}</td>".format(color_info)

			#html += "<td>{}</td>".format("<a href=\"" + v['link'] + "\"> [" + v['year'] + '] ' + v['title'])
			#html += "<td>{}</td>".format(v['link'])
			#html += "<td>{}</td>".format(v['file'])
			html += "</tr>"
			counter += 1
		html += "</table></html>"
		
		file.write(html)

	
def get_entity(gene_ncbi_id, entities):
	ret_val = []
	for entity in entities:
		if entity[0] == gene_ncbi_id:
			ret_val.append(entity)
	return ret_val


import random
import colorsys

predefined_colors = ['#8AFF9C', '#7AFBFF', '#FF80C5', '#FF9E9E', '#91AAFF', '#FF00FF', '#00FEFE', '#FFFF00', '#00FF00']  # predefined nice colors for highlight
existing_colors = set()


def reset_color_values():
	global predefined_colors
	global existing_colors
	predefined_colors = ['#8AFF9C', '#7AFBFF', '#FF80C5', '#FF9E9E', '#91AAFF', '#FF00FF', '#00FEFE', '#FFFF00', '#00FF00']  # predefined nice colors for highlight
	existing_colors = set()
	
	
def random_color():
	"""
	ref: https://stackoverflow.com/questions/43437309/get-a-bright-random-colour-python
	"""
	if len(predefined_colors) > 0:
		color = predefined_colors.pop()
	else:	
		h,s,l = random.random(), 0.7 + random.random()/2.0, 0.4 + random.random()/5.0
		r,g,b = [int(256*i) for i in colorsys.hls_to_rgb(h,l,s)]
		color = '#%02X%02X%02X' % (r,g,b)
		
		if color == '#FFFFFF':
			print('white')
			input()

	return color if color not in existing_colors else random_color()


def highlight_entities(html_text, entities):
	extra_offset = 0
	
	for entity in entities:
		#print('entity:', entity)
		#print('Before:', html_text)

		text = entity[0]
		offset = entity[1]
		length = entity[2]
		color = entity[3]
		
		html_text = html_text[:offset + extra_offset] + '<span style="background-color:' + color + '">' + html_text[offset + extra_offset:]
		extra_offset += len('<span style="background-color:' + color + '">')
		
		html_text = html_text[:offset + length + extra_offset] + '</span>' + html_text[offset + length + extra_offset:]
		extra_offset += len('</span>')
		#print('---------------------------------------')
		#print('After:', html_text)

	return html_text


def remove_unnecessary_token(psg_data):
	"""
	remove unnecessary tokens. '-' causes an error, so remove it.
	"""
	# debugging
	#debug_flag = False
	#print('Before:', psg_data)
	
	for tok_idx, tok in enumerate(psg_data['parsed_text']):
		if tok == '-':
			for x in range(len(psg_data['entities'])): # (ent_ncbi_id, ent_text, ent_type, ent_start, ent_end, offset, length)
				if tok_idx < psg_data['entities'][x][3]:
					psg_data['entities'][x][3] -= 1
					psg_data['entities'][x][4] -= 1
					
					#debug_flag = True
				elif tok_idx > psg_data['entities'][x][3] and tok_idx < psg_data['entities'][x][4]:
					psg_data['entities'][x][4] -= 1
					
					#debug_flag = True
	
	psg_data['parsed_text'] = [tok.replace('-', '') for tok in psg_data['parsed_text'] if tok != '-']		
	
	for x in range(len(psg_data['entities'])): # (ent_ncbi_id, ent_text, ent_type, ent_start, ent_end, offset, length)
		psg_data['entities'][x][1] = psg_data['entities'][x][1].replace('-', ' ')
		psg_data['entities'][x] = tuple(psg_data['entities'][x])
	
	# debugging			
	#print('----------------------------------------------------')
	#print('After :', psg_data)
	#if debug_flag:
	#	input()
	

def run():
	global predefined_colors
	
	json_file = '/direct/sdcc+u/gpark/BER-NLP/BERT-Relation-Extraction/data/PMtask_Relations_TrainingSet.json'
	#json_file = '/direct/sdcc+u/gpark/BER-NLP/BERT-Relation-Extraction/data/PMtask_Relation_TestSet.json'

	with open(json_file) as fp:
		data = json.load(fp)

	nlp = spacy.load("en_core_web_lg", disable=["tagger", "ner"])

	loc_more_than_2 = 0 # debug

	D = []
	text_for_html = []
	for doc in data["documents"]:
		
		# TODO: fix the highlight error this document in test set.
		doc_id = doc["id"]
		if doc_id == "14645919":
			continue
		
		psg_data_list = []
		for psg in doc["passages"]:
			psg_data = {}
			
			psg_text = psg["text"]
			psg_offset = psg["offset"]
			
			psg_data['raw_text'] = psg_text

			psg_tokens = nlp(psg_text)
			
			
			# TODO: remove this after testing - consider only a single sentence passage.
			if len(list(psg_tokens.sents)) > 1:
				#print(psg_text)
				#print(list(psg_tokens.sents))
				continue
				

			entities = []
			for anno in psg["annotations"]:
				ent_text = anno["text"]
				ent_type = anno["infons"]["type"]
				
				if json_file.endswith('TestSet.json'):
					ent_ncbi_id = anno["infons"]["identifier"]
				else:
					ent_ncbi_id = anno["infons"]["NCBI GENE"]
				
				# debug
				if len(anno["infons"]) > 2:
					print('more than 3 values in infons')
					for i in anno["infons"]:
						print(i)
					input()
				  
				# TODO: handle entities constructed from several tokens. they are few (#21), so ignore them for now.
				"""
				e.g., 
					tokens: ... 'nesprin-1', 'and', '-2', ...
					entity: 'nesprin-2'
						"locations": [
							{
							  "length": 7, 
							  "offset": 203
							}, 
							{
							  "length": 2, 
							  "offset": 217
							}
					
				"""
				if len(anno["locations"]) > 1:
					loc_more_than_2 += 1 # debug
					continue

				for loc in anno["locations"]:
					length = loc["length"]
					offset = loc["offset"] - psg_offset
					
					ent_start = ent_end = -1
					
					idx_error = False
					
					for tok_seq, tok in enumerate(psg_tokens):
						if tok.idx == offset:
							ent_start = tok_seq
							ent_end = tok_seq + 1
							
							if len(ent_text) > len(tok.text):
								"""
								e.g.,
									token: beta
									entity: beta-tublin
								"""
								for x in range(tok_seq + 1, len(psg_tokens)):
									if ent_text.endswith(psg_tokens[x].text):
										ent_end = x + 1
										break
									
									if psg_tokens[x].text not in ent_text:
										"""
										e.g., 
											psg_tokens: ['The', 'DEAD', '-', 'box', 'helicase', 'DDX3X', 'is', 'a', 'critical', 'component', 'of', 'the', 'TANK', '-', 'binding', 'kinase', '1-dependent', 'innate', 'immune', 'response', '.']                                                                                       │11/16/2020 09:14:45 AM [INFO]: Last batch samples (pos, neg): 1, 16
											tok.text: TANK
											ent_text: TANK-binding kinase 1 
											
										there is an index errors in the file.
										the error, the offset 339 - 99 (passage offset) = 240 that indicates '.' in the tokens, ... 'receptor', 'R2', '.', 'Crystals', ...
											{
											  "text": "receptor R2", 
											  "infons": {
												"type": "Gene", 
												"NCBI GENE": "7133"
											  }, 
											  "id": "15", 
											  "locations": [
												{
												  "length": 11, 
												  "offset": 339
												}
											  ]
											}
										"""
										ent_end_pos = offset + length - psg_tokens[x].idx
										
										ent_segment = psg_tokens[x].text[:ent_end_pos]
										tail_text = psg_tokens[x].text[ent_end_pos:]
																			
										if ent_segment not in ent_text:	# index error in the file.
											idx_error = True
											break
										
										# debug
										'''
										print('ent_segment:', ent_segment)
										print('tail_text:', tail_text)
										print("ent_text:", ent_text, '/ offset:', offset)
										print("psg_tokens[x].text:", psg_tokens[x].text, '/ psg_tokens[x].idx:', psg_tokens[x].idx)
										print("Before:", [x.text for x in psg_tokens])
										'''

										with psg_tokens.retokenize() as retokenizer:
											heads = [(psg_tokens[x], 1), psg_tokens[x - 1]]
											retokenizer.split(psg_tokens[x], [ent_segment, tail_text], heads=heads)
										
										ent_end = x + 1
										
										#print("After:", [x.text for x in psg_tokens])
										
										break
		
							elif len(ent_text) < len(tok.text):
								"""
								e.g.,
									token: sin3-binding
									entity: sin3
									-> split token into 'sin3', '-binding' (tail_text)
								"""
								
								# debug
								'''
								print("2 ent_text:", ent_text, '/ offset:', offset)
								print("2 tok.text:", tok.text, '/ tok.idx:', tok.idx)
								print("2 tok_seq:", tok_seq)
								print("2 Before:", [x.text for x in psg_tokens])
								'''
								
								with psg_tokens.retokenize() as retokenizer:
									heads = [(psg_tokens[tok_seq], 1), psg_tokens[tok_seq - 1]]
									tail_text = tok.text.split(ent_text, 1)[1]
									retokenizer.split(psg_tokens[tok_seq], [ent_text, tail_text], heads=heads)

								#print("2 After:", [x.text for x in psg_tokens])

							break
					
					if idx_error == False and ent_start != -1 and ent_end != -1:
						#entity_annotation = (ent_ncbi_id, ent_text, ent_type, ent_start, ent_end, offset, length)
						entity_annotation = [ent_ncbi_id, ent_text, ent_type, ent_start, ent_end, offset, length]
						entities.append(entity_annotation)					
						
						#print('entity annotation:', ent_ncbi_id, ent_text, ent_type, ent_start, ent_end, offset, length)
			
			psg_data['parsed_text'] = [tok.text for tok in psg_tokens]
			psg_data['entities'] = entities

			remove_unnecessary_token(psg_data) # remove '-' that causes an error in BERT Relation Extraction.

			psg_data_list.append(psg_data)

		for rel in doc["relations"]:
			rel_id = rel["id"]
			gene1_ncbi_id = rel["infons"]["Gene1"]
			gene2_ncbi_id = rel["infons"]["Gene2"]
			gene_rel = rel["infons"]["relation"]
			
			for psg_data in psg_data_list:
				gene1_entities = get_entity(gene1_ncbi_id, psg_data['entities'])
				gene2_entities = get_entity(gene2_ncbi_id, psg_data['entities'])

				for g1e in gene1_entities:
					for g2e in gene2_entities:
						parsed_text = psg_data['parsed_text']
						
						g1_text = g1e[1]
						g1_s = g1e[3]
						g1_e = g1e[4]
						g2_text = g2e[1]
						g2_s = g2e[3]
						g2_e = g2e[4]
						
						D.append((
								 ((parsed_text, (g1_s, g1_e), (g2_s, g2_e))),
								 g1_text, 
								 g2_text
								 ))
						
						# debug
						'''
						print(psg_data['parsed_text'], g1e[3:5], g2e[3:5], g1e[1], g2e[1])

						if g1e[1].startswith(parsed_text[g1_s]) == False or g1e[1].endswith(parsed_text[g1_e - 1]) == False or \
						   g2e[1].startswith(parsed_text[g2_s]) == False or g2e[1].endswith(parsed_text[g2_e - 1]) == False:
							print(psg_data['parsed_text'][g1_s:g1_e])
							print(psg_data['parsed_text'][g2_s:g2_e])
							print('Mismatch error!!')
							input()
						'''
		
		# sample visualization in html
		for psg_data in psg_data_list:
			html_info = {}
			html_info['color_info'] = []
			
			all_entities = {}
			rel_pairs = []
			for rel in doc["relations"]:
				rel_id = rel["id"]
				gene1_ncbi_id = rel["infons"]["Gene1"]
				gene2_ncbi_id = rel["infons"]["Gene2"]
				gene_rel = rel["infons"]["relation"]
				
				gene1_entities = all_entities.get(gene1_ncbi_id) if all_entities.get(gene1_ncbi_id) != None else get_entity(gene1_ncbi_id, psg_data['entities'])
				gene2_entities = all_entities.get(gene2_ncbi_id) if all_entities.get(gene2_ncbi_id) != None else get_entity(gene2_ncbi_id, psg_data['entities'])
				
				if gene1_ncbi_id not in all_entities:
					gene1_color = random_color()
					existing_colors.add(gene1_color)
					all_entities[gene1_ncbi_id] = [(e[1], e[5], e[6], gene1_color) for e in gene1_entities]
				if gene2_ncbi_id not in all_entities:
					gene2_color = random_color()
					existing_colors.add(gene2_color)
					all_entities[gene2_ncbi_id] = [(e[1], e[5], e[6], gene2_color) for e in gene2_entities]
				
				if len(gene1_entities) > 0 and len(gene2_entities) > 0:
					rel_pairs += all_entities[gene1_ncbi_id]
					rel_pairs += all_entities[gene2_ncbi_id]

					html_info['color_info'].append((gene1_ncbi_id, all_entities[gene1_ncbi_id][0][3], gene2_ncbi_id, all_entities[gene2_ncbi_id][0][3]))
			
			if len(rel_pairs) > 0:
				rel_pairs = list(set(rel_pairs))
				rel_pairs.sort(key=lambda rel_pairs: rel_pairs[1])	# sort by offset
				html_info['html_text'] = highlight_entities(psg_data['raw_text'], rel_pairs)
				text_for_html.append(html_info)
				
			reset_color_values()

	#write_to_file(text_for_html, 'BioCreative_VI_Relation_Extraction_training_set.html')

	#print('loc_more_than_2:', loc_more_than_2)


def main():
	run()


if __name__ == "__main__":
    main()
		