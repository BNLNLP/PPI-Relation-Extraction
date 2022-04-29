"""
This code converts the ChemProt format from Biomedical Language Understanding and Reasoning Benchmark (BLURB) to the format that fits to the model. 03-18-2022

- ChemProt has a single label for a sentence.
- ChemProt has samples with overlapping entity (CHEM-GENE).

ref: https://microsoft.github.io/BLURB/tasks.html#dataset_chemprot

"""
import os
import csv
import json


current_working_dir = os.getcwd()

data_dir = os.path.join(current_working_dir, 'datasets/data_generation/data/chemprot (entity dummification and entity types)')
#data_dir = os.path.join(current_working_dir, 'datasets/data_generation/data/chemprot (entity dummification)')
#data_dir = os.path.join(current_working_dir, 'datasets/data_generation/data/chemprot (original)')

out_dir = os.path.join(current_working_dir, 'datasets/ChemProt_BLURB')

entity_type_file = os.path.join(current_working_dir, 'datasets/ChemProt_BLURB/entity_types.json')
entity_types = json.load(open(entity_type_file))

relation_type_file = os.path.join(current_working_dir, 'datasets/ChemProt_BLURB/relation_types.json')
relation_types = json.load(open(relation_type_file))

# debug
total_num_of_samples = 0

for filename in os.listdir(data_dir):
    f = open(os.path.join(data_dir, filename))
    tsv_reader = csv.reader(f, delimiter="\t")

    output_txt = ''
    for row in tsv_reader:
        index = row[0]
        text = row[1]
        label = row[2]

        text = row[1].rsplit("%%", 2)
        text, entity_1_type, entity_2_type = text[0], text[1].upper(), text[2].upper()

        if text.count("@GENE$") == 1 and text.count("@CHEMICAL$") == 1:
            if text.index('@GENE$') < text.index('@CHEMICAL$'):
                e1_start = text.index('@GENE$')
                e1_end   = text.index('@GENE$') + len('@GENE$')
                e2_start = text.index('@CHEMICAL$')
                e2_end   = text.index('@CHEMICAL$') + len('@CHEMICAL$')
            else:
                e2_start = text.index('@GENE$')
                e2_end   = text.index('@GENE$') + len('@GENE$')
                e1_start = text.index('@CHEMICAL$')
                e1_end   = text.index('@CHEMICAL$') + len('@CHEMICAL$')
        # CHEM-GENE is used to tag the overlapping entity.
        elif text.count("@CHEM-GENE$") == 1:
            e1_start = e2_start = text.index("@CHEM-GENE$")
            e1_end   = e2_end   = text.index("@CHEM-GENE$") + len("@CHEM-GENE$")
        else:
            raise Exception("No tags in the sentence: " f"{text}.")
        
        entity_1 = text[e1_start:e1_end]
        entity_2 = text[e2_start:e2_end]
        
        # Add entity markers.
        text_with_entity_marker = text
        text_with_typed_entity_marker = text
        
        ### TODO: test this and compare with the origianl types.
        entity_1_type = entity_1[1:-1]
        entity_2_type = entity_2[1:-1]
        
        e1_typed_marker_s, e1_typed_marker_e = '[' + entity_1_type + ']', '[/' + entity_1_type + ']'
        e2_typed_marker_s, e2_typed_marker_e = '[' + entity_2_type + ']', '[/' + entity_2_type + ']'
        
        if text.count("@CHEM-GENE$") == 1:
            text_with_entity_marker = text_with_entity_marker.replace("@CHEM-GENE$", "[E1-E2]@CHEM-GENE$[/E1-E2]")
            e1_s_in_text_with_entity_marker = e2_s_in_text_with_entity_marker = text_with_entity_marker.index('[E1-E2]') + len('[E1-E2]')
            e1_e_in_text_with_entity_marker = e2_e_in_text_with_entity_marker = text_with_entity_marker.index('[/E1-E2]')
            
            text_with_typed_entity_marker = text_with_typed_entity_marker.replace("@CHEM-GENE$", "[CHEM-GENE]@CHEM-GENE$[/CHEM-GENE]")
            e1_s_in_text_with_typed_entity_marker = e2_s_in_text_with_typed_entity_marker = text_with_typed_entity_marker.index('[CHEM-GENE]') + len('[CHEM-GENE]')
            e1_e_in_text_with_typed_entity_marker = e2_e_in_text_with_typed_entity_marker = text_with_typed_entity_marker.index('[/CHEM-GENE]')
        else:
            text_with_entity_marker = text_with_entity_marker.replace(entity_1, '[E1]' + entity_1 + '[/E1]')
            text_with_entity_marker = text_with_entity_marker.replace(entity_2, '[E2]' + entity_2 + '[/E2]')
            e1_s_in_text_with_entity_marker = text_with_entity_marker.index('[E1]') + len('[E1]')
            e1_e_in_text_with_entity_marker = text_with_entity_marker.index('[/E1]')
            e2_s_in_text_with_entity_marker = text_with_entity_marker.index('[E2]') + len('[E2]')
            e2_e_in_text_with_entity_marker = text_with_entity_marker.index('[/E2]')
            
            # don't use replace() since two typed markers can be the same.
            text_with_typed_entity_marker = text_with_typed_entity_marker[:e1_start] + \
                                            e1_typed_marker_s + entity_1 + e1_typed_marker_e + \
                                            text_with_typed_entity_marker[e1_end:e2_start] + \
                                            e2_typed_marker_s + entity_2 + e2_typed_marker_e + \
                                            text_with_typed_entity_marker[e2_end:]
                                   
            # don't use index() because two typed markers can be the same.
            e1_s_in_text_with_typed_entity_marker = e1_start + len(e1_typed_marker_s)
            e1_e_in_text_with_typed_entity_marker = e1_end + len(e1_typed_marker_s)
            e2_s_in_text_with_typed_entity_marker = e2_start + len(e1_typed_marker_s) + len(e1_typed_marker_e) + len(e2_typed_marker_s)
            e2_e_in_text_with_typed_entity_marker = e2_end + len(e1_typed_marker_s) + len(e1_typed_marker_e) + len(e2_typed_marker_s)
        
        # debug
        if entity_1 != text_with_entity_marker[e1_s_in_text_with_entity_marker:e1_e_in_text_with_entity_marker] or \
           entity_1 != text_with_typed_entity_marker[e1_s_in_text_with_typed_entity_marker:e1_e_in_text_with_typed_entity_marker] or \
           entity_2 != text_with_entity_marker[e2_s_in_text_with_entity_marker:e2_e_in_text_with_entity_marker] or \
           entity_2 != text_with_typed_entity_marker[e2_s_in_text_with_typed_entity_marker:e2_e_in_text_with_typed_entity_marker]:
            print('text:', text)
            print('entity_1:', entity_1, '/ entity_2:', entity_2)
            print('text_with_entity_marker:', text_with_entity_marker)
            print('e1_s_in_text_with_entity_marker:', e1_s_in_text_with_entity_marker, '/ e1_e_in_text_with_entity_marker:', e1_e_in_text_with_entity_marker)
            print('e2_s_in_text_with_entity_marker:', e2_s_in_text_with_entity_marker, '/ e2_e_in_text_with_entity_marker:', e2_e_in_text_with_entity_marker)
            print(text_with_entity_marker[e1_s_in_text_with_entity_marker:e1_e_in_text_with_entity_marker])
            print(text_with_entity_marker[e2_s_in_text_with_entity_marker:e2_e_in_text_with_entity_marker])
            print('text_with_typed_entity_marker:', text_with_typed_entity_marker)
            print('e1_s_in_text_with_typed_entity_marker:', e1_s_in_text_with_typed_entity_marker, '/ e1_e_in_text_with_typed_entity_marker:', e1_e_in_text_with_typed_entity_marker)
            print('e2_s_in_text_with_typed_entity_marker:', e2_s_in_text_with_typed_entity_marker, '/ e2_e_in_text_with_typed_entity_marker:', e2_e_in_text_with_typed_entity_marker)
            print(text_with_typed_entity_marker[e1_s_in_text_with_typed_entity_marker:e1_e_in_text_with_typed_entity_marker])
            print(text_with_typed_entity_marker[e2_s_in_text_with_typed_entity_marker:e2_e_in_text_with_typed_entity_marker])
            input('enter...')
            
        relation = {'relation_type': 'CPR:false' if label == 'false' else label, 
                    'relation_id': relation_types['CPR:false']['id'] if label == 'false' else relation_types[label]['id'],
                    'entity_1': entity_1,
                    'entity_1_idx': (e1_start, e1_end),
                    'entity_1_idx_in_text_with_entity_marker': (e1_s_in_text_with_entity_marker, e1_e_in_text_with_entity_marker),
                    'entity_1_idx_in_text_with_typed_entity_marker': (e1_s_in_text_with_typed_entity_marker, e1_e_in_text_with_typed_entity_marker),
                    'entity_1_type': entity_1_type,
                    'entity_1_type_id': entity_types[entity_1_type]['id'],
                    'entity_2': entity_2,
                    'entity_2_idx': (e2_start, e2_end),
                    'entity_2_idx_in_text_with_entity_marker': (e2_s_in_text_with_entity_marker, e2_e_in_text_with_entity_marker),
                    'entity_2_idx_in_text_with_typed_entity_marker': (e2_s_in_text_with_typed_entity_marker, e2_e_in_text_with_typed_entity_marker),
                    'entity_2_type': entity_2_type, 
                    'entity_2_type_id': entity_types[entity_2_type]['id']}
        
        # 'relation' item indicates the relation directionality. a.k.a symmetric or asymmetric relation.
        # 'reverse' item is only used for undirected relations. 
        # ('reverse' item) For testing phase, undirected samples are replicated, and the replicated samples are tagged as reverse. 
        # ('reverse' item) So, if it's set to true, the model uses the second entity + the first entity instead of 
        # ('reverse' item) the first entity + the second entity to classify both relation representation cases (A + B, B + A). 
        output_txt += json.dumps({"id": filename.split('.')[0] + '_' + str(index),
                                  "text": text,
                                  "text_with_entity_marker": text_with_entity_marker,
                                  "text_with_typed_entity_marker": text_with_typed_entity_marker,
                                  "relation": [relation],
                                  #"directed": True,
                                  #"reverse": False,
                                  })
        output_txt += '\n'
        
        total_num_of_samples += 1

    outfile = os.path.join(out_dir, filename.replace('.tsv', '_0.json'))
    with open(outfile, "w") as f:
        f.write(output_txt)

# debug
print('>> total_num_of_samples:', total_num_of_samples)

    