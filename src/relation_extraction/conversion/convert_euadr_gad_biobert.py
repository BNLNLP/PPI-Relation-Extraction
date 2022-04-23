"""
This code converts the EU-ADR/GAD format from BioBERT to the format that fits to the model. 01-13-2022

- Tha dataset is 10-fold CV.
- EU-ADR/GAD has a single label for a sentence.

ref: https://drive.google.com/file/d/1-jDKGcXREb2X9xTFnuiJ36PvsqoyHWcw/view?usp=drive_open

"""
import os
import csv
import json


dataset_name = globals()['dataset_name']
current_working_dir = os.getcwd()

data_dir = os.path.join(current_working_dir, 'datasets/' + dataset_name + '/biobert_10-fold_CV')
out_dir = os.path.join(current_working_dir, 'datasets/' + dataset_name)

entity_type_file = os.path.join(current_working_dir, 'datasets/' + dataset_name + '/entity_types.json')
entity_types = json.load(open(entity_type_file))

# debug
num_of_total_samples = 0

for root, dirs, files in os.walk(data_dir):
    for filename in files:
        if filename == 'dev.tsv': # it's a CV dataset, so the dev file is empty.
            continue
        
        print(os.path.join(root, filename))
        
        cv_num = str(int(root.rsplit('/', 1)[1])-1) # change the num to start from 0 instead of 1.
        
        f = open(os.path.join(root, filename))
        
        tsv_reader = csv.reader(f, delimiter="\t")
        
        if filename == 'test.tsv': # test.tsv and train.tsv have different formats.
            next(tsv_reader) # skip the first row of the TSV file.
        
        output_txt = ''
        for row_num, row in enumerate(tsv_reader):
            if filename == 'test.tsv':  # test.tsv and train.tsv have different formats.
                index = row[0]
                text = row[1]
                label = row[2]
            elif filename == 'train.tsv':
                index = row_num
                text = row[0]
                label = row[1]

            if text.count("@GENE$") != 1 and text.count("@DISEASE$") != 1:
                raise Exception("No tags in the sentence: " f"{text}.")

            if text.index('@GENE$') < text.index('@DISEASE$'):
                e1_start = text.index('@GENE$')
                e1_end   = text.index('@GENE$') + len('@GENE$')
                e2_start = text.index('@DISEASE$')
                e2_end   = text.index('@DISEASE$') + len('@DISEASE$')
            else:
                e2_start = text.index('@GENE$')
                e2_end   = text.index('@GENE$') + len('@GENE$')
                e1_start = text.index('@DISEASE$')
                e1_end   = text.index('@DISEASE$') + len('@DISEASE$')

            entity_1 = text[e1_start:e1_end]
            entity_2 = text[e2_start:e2_end]

            # Add entity markers.
            text_with_entity_marker = text
            text_with_typed_entity_marker = text
            
            entity_1_type = entity_1[1:-1]
            entity_2_type = entity_2[1:-1]
            
            e1_typed_marker_s, e1_typed_marker_e = '[' + entity_1_type + ']', '[/' + entity_1_type + ']'
            e2_typed_marker_s, e2_typed_marker_e = '[' + entity_2_type + ']', '[/' + entity_2_type + ']'
            
            text_with_entity_marker = text_with_entity_marker.replace(entity_1, '[E1]' + entity_1 + '[/E1]')
            text_with_entity_marker = text_with_entity_marker.replace(entity_2, '[E2]' + entity_2 + '[/E2]')
            e1_s_in_text_with_entity_marker = text_with_entity_marker.index('[E1]') + len('[E1]')
            e1_e_in_text_with_entity_marker = text_with_entity_marker.index('[/E1]')
            e2_s_in_text_with_entity_marker = text_with_entity_marker.index('[E2]') + len('[E2]')
            e2_e_in_text_with_entity_marker = text_with_entity_marker.index('[/E2]')

            text_with_typed_entity_marker = text_with_typed_entity_marker.replace(entity_1, e1_typed_marker_s + entity_1 + e1_typed_marker_e)
            text_with_typed_entity_marker = text_with_typed_entity_marker.replace(entity_2, e2_typed_marker_s + entity_2 + e2_typed_marker_e)		
            e1_s_in_text_with_typed_entity_marker = text_with_typed_entity_marker.index(e1_typed_marker_s) + len(e1_typed_marker_s)
            e1_e_in_text_with_typed_entity_marker = text_with_typed_entity_marker.index(e1_typed_marker_e)
            e2_s_in_text_with_typed_entity_marker = text_with_typed_entity_marker.index(e2_typed_marker_s) + len(e2_typed_marker_s)
            e2_e_in_text_with_typed_entity_marker = text_with_typed_entity_marker.index(e2_typed_marker_e)

            relation = {'relation_type': 'positive' if label == '1' else 'negative', 
                        'relation_id': int(label), 
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
            output_txt += json.dumps({"id": filename.split('.')[0] + '_' + cv_num + '_' + str(index),
                                      "text": text,
                                      "text_with_entity_marker": text_with_entity_marker,
                                      "text_with_typed_entity_marker": text_with_typed_entity_marker,
                                      "relation": [relation],
                                      #"directed": True,
                                      #"reverse": False,
                                      })
            output_txt += '\n'
            
            num_of_total_samples += 1

        outfile = os.path.join(out_dir, filename.replace('.tsv', '_' + cv_num +'.json'))
        with open(outfile, "w") as f:
            f.write(output_txt)

# debug
print('>> num_of_total_samples:', num_of_total_samples)

    