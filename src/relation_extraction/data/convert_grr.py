"""
This generates 10 fold cross-validation sets for GRR (Gene regulatory relation) datasets.

- Annotator: Sean

"""
import os
import json
import re
import logging
import itertools
import numpy as np

from sklearn.model_selection import KFold, StratifiedKFold

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')


current_working_dir = os.getcwd()
    
#data_file_dict = {"P-putida": "datasets/GRR/P-putida/curated_putitda_abstract_text.annot.1-1100.json"}
#data_file_dict = {"P-putida": "datasets/GRR/P-aeruginosa & P-fluorescens/curated_positive_pairs_annot.json"}
data_file_dict = {"P-putida": "datasets/GRR/P-species/combined_P-species_annot.json"}

entity_type_file = os.path.join(current_working_dir, 'datasets/GRR/P-putida/entity_types.json')
entity_types = json.load(open(entity_type_file))

relation_type_file = os.path.join(current_working_dir, 'datasets/GRR/P-putida/relation_types.json')
relation_types = json.load(open(relation_type_file))

'''
# detailed relation types
relation_type_dict = {'regulate': ['regulat', 'regulate'],
                      'upregulate': ['upregulat', 'upregulate'], 
                      'downregulate': ['downregulat', 'downregulate'], 
                      'activate': ['activat', 'activate'], 
                      'inactivate': ['inactivat', 'inactivate'], 
                      'modulate': ['modulat', 'modulate'], 
                     }
'''
relation_type_dict = {'regulate': ['regulat', 'regulate', 'control', 'modulat', 'modulate'],
                      'negative_regulate': ['inhibit', 'inactivat', 'inactivate', 'downregulat', 'downregulate', 'repress', 'suppress'], 
                      'positive_regulate': ['activat', 'activate', 'upregulat', 'upregulate'], 
                      'interact': ['interact', 'bind'], 
                     }


def get_samples(file):
    data = json.load(open(file))
    
    
    '''
    Annot=1 should have correct values for
     "correct_rel"  - my final call for relationship
      "correct_genes" - my final call for the actual genes
      "full_text_needed" - y or n for whether the information comes from text outside the single sentence
    
    {
        'orig': {
            'sent_id': 'downregulat_14',
            'paper': 'PMC7461776',
            'sent_with_neighboring_text_tokens': [['If', 0], ['there', 1], ['is', 2], ['no', 3], ['excessive', 4], ['accumulation', 5], ['of', 6], ['iron', 7], ['in', 8], ['cells', 9], [',', 10], ['how', 11], ['can', 12], ['the', 13], ['downregulation', 14], ['of', 15], ['pyoverdine', 16], ['-', 17], ['related', 18], ['genes', 19], ['and', 20], ['the', 21], ['fpvA', 22], ['gene', 23], ['be', 24], ['explained', 25], ['?', 26], ['As', 27], ['an', 28], ['alternative', 29], [',', 30], ['we', 31], ['hypothesized', 32], ['that', 33], ['Bip', 34], ['interferes', 35], ['with', 36], ['the', 37], ['ion', 38], ['homeostasis', 39], ['of', 40], ['cells', 41], ['by', 42], ['chelating', 43], ['intracellular', 44], ['iron', 45], ['ions', 46], ['or', 47], ['other', 48], ['metal', 49], ['ions', 50], ['.', 51], ['In', 52], ['fact', 53], [',', 54], ['Bip', 55], ['binds', 56], ['not', 57], ['only', 58], ['iron', 59], ['with', 60], ['high', 61], ['affinity', 62], ['(', 63], ['Fe3+', 64], [':', 65], ['ΔG', 66], ['=', 67], ['−', 68], ['137', 69], ['kJ', 70], ['mol−1', 71], [',', 72], ['Fe2+', 73], [':', 74], ['ΔG', 75], ['=', 76], ['−', 77], ['69', 78], ['kJ', 79], ['mol−1', 80], [')', 81], ['but', 82], ['also', 83], ['copper', 84], ['ions', 85], ['(', 86], ['Cu2+', 87], [':', 88], ['ΔG', 89], ['=', 90], ['−', 91], ['120', 92], ['kJ', 93], ['mol−1', 94], [')', 95], [',', 96], ['and', 97], ['when', 98], ['accumulated', 99], ['in', 100], ['cells', 101], [',', 102], ['it', 103], ['may', 104], ['compete', 105], ['with', 106], ['cellular', 107], ['proteins', 108], ['for', 109], ['these', 110], ['molecules', 111], ['.', 112]],
            'sent_with_neighboring_text': 'If there is no excessive accumulation of iron in cells, how can the downregulation of pyoverdine-related genes and the fpvA gene be explained? As an alternative, we hypothesized that Bip interferes with the ion homeostasis of cells by chelating intracellular iron ions or other metal ions. In fact, Bip binds not only iron with high affinity (Fe3+: ΔG = −137 kJ mol−1, Fe2+: ΔG = −69 kJ mol−1) but also copper ions (Cu2+: ΔG = −120 kJ mol−1), and when accumulated in cells, it may compete with cellular proteins for these molecules .',
            'text_no': 1283,
            'sent': 'If there is no excessive accumulation of iron in cells, how can the downregulation of pyoverdine-related genes and the fpvA gene be explained?',
            'genes': [['fpvA', 22]]
        },
        'oper': '',
        'rel': 'downregulat',
        'comment': 'weak statement of inhibition?',
        'annot': '1',
        'correct_rel': 'inhibit',
        'auto_eval': ['UniqueSingular', 'Singular'],
        'correct': 'x',
        'full_text_needed': 'y',
        'correct_genes': [['Bip', 'fpvA']],
        'genes': ['fpvA'],
        'sample_no': 1283,
        'text_no': 1283
    }
    '''
    samples = {}
    # debug
    relation_cnt = {}
    
    for item in data:
        # annot = 1, which are positives PPI causal statements. annot = 2-5, or x should be considered negative.
        if item['annot'] == '1':
            relation_type = item['correct_rel'][0] if isinstance(item['correct_rel'], list) else item['correct_rel']
            gene_pairs = item['correct_genes']
            neighboring_txt_needed = True if item['full_text_needed'] == 'y' else False
            text = item['orig']['sent_with_neighboring_text'] if neighboring_txt_needed else item['orig']['sent']
            sample_no = item['sample_no']
            doc_id = item['orig']['paper']
            
            
            
            
            
            
            if text.startswith('The switch between the TCA cycle and glyoxylate shunt is controlled'):
                continue
            
            
            
            
            
            
            # get a standardized relation form.
            for k, v in relation_type_dict.items():
                if relation_type in v:
                    relation_type = k
                    break

            # debug
            if relation_type not in relation_types.keys():
                print('Unknown relation - ', relation_type, '/ sample_no:', sample_no)
                input('enter..')
                continue

            for g_p in gene_pairs:
                # debug
                if len(g_p) != 2:
                    print('sample_no:', sample_no, '/ gene pair:', g_p)
                    input('enter..')
                    continue
                    
                e1_text, e2_text = g_p
                
                # debug
                #if '/' in e1_text or '/' in e2_text:
                #	print('e1_text:', e1_text, '| e2_text:', e2_text)
                #	input('enter..')
                
                # e.g., rhlAB(C)
                e1_text = e1_text.replace('(', '\(').replace(')', '\)')  
                e2_text = e2_text.replace('(', '\(').replace(')', '\)')
                
                def is_alpha(ch):
                    return True if ((ch >='a' and ch <= 'z') or (ch >='A' and ch <='Z')) else False
                
                e1_indice = []
                for m in re.finditer(e1_text, text):
                    prev_ch = text[m.start(0)-1] if m.start(0) != 0 else '1'
                    next_ch = text[m.end(0)] if m.end(0) != len(text) else '1'
                    
                    if is_alpha(prev_ch) is False and is_alpha(next_ch) is False:
                        e1_indice.append((m.start(0), m.end(0)))
                    '''
                    if re.search('[a-zA-Z]', text[m.start(0)-1]) or re.search('[a-zA-Z]', text[m.end(0)]):
                        print(text)
                        print(e1_text, m.start(0), m.end(0))
                        print(text[m.start(0)-1:m.end(0)+1])
                        input('enter..')
                    '''
                e2_indice = []
                for m in re.finditer(e2_text, text):
                    prev_ch = text[m.start(0)-1] if m.start(0) != 0 else '1'
                    next_ch = text[m.end(0)] if m.end(0) != len(text) else '1'
                    
                    if is_alpha(prev_ch) is False and is_alpha(next_ch) is False:
                        e2_indice.append((m.start(0), m.end(0)))	

                #e1_indice = [(m.start(0), m.end(0)) for m in re.finditer(e1_text, text)]
                #e2_indice = [(m.start(0), m.end(0)) for m in re.finditer(e2_text, text)]
                
                '''
                if len(e1_indice) > 1:
                    for i in e1_indice:
                        print('e1_indice:', i, text[i[0]:i[1]])
                    #input('enter..')
                    
                if len(e2_indice) > 1:
                    for i in e2_indice:
                        print('e2_indice:', i, text[i[0]:i[1]])
                    #input('enter..')
                '''	
                
                # debug
                if len(e1_indice) == 0 or len(e2_indice) == 0:
                    print('sample_no:', sample_no, '/ gene pair:', g_p)
                    print('e1_indice:', e1_indice, '| e2_indice:', e2_indice)
                    input('enter..')
                
                # debug
                '''
                for (x, y) in list(itertools.product(e1_indice, e2_indice)):
                    if x == y:
                        print(x, y)
                        input('enter..')
                '''
                
                all_combinations = [(x, y) for (x, y) in list(itertools.product(e1_indice, e2_indice)) if x != y]
                
                for pair_no, (e1_idx, e2_idx) in enumerate(all_combinations, 1):
                    # place e1, e2 by sequence order.
                    if e1_idx[0] < e2_idx[0]:
                        e1_start_idx, e1_end_idx = e1_idx
                        e2_start_idx, e2_end_idx = e2_idx
                        e1_text, e2_text = g_p
                    else:
                        e1_start_idx, e1_end_idx = e2_idx
                        e2_start_idx, e2_end_idx = e1_idx
                        e1_text, e2_text = g_p[::-1]
                    
                    e1_type = e2_type = 'gene'
                    
                    # debug
                    if text[e1_start_idx:e1_end_idx] != e1_text or text[e2_start_idx:e2_end_idx] != e2_text:
                        raise Exception("ERROR - entity text mismatch: " f"{text}.") 
                    
                    if e1_start_idx == e2_start_idx \
                       or (e1_start_idx > e2_start_idx and e1_start_idx < e2_end_idx) \
                       or (e2_start_idx > e1_start_idx and e2_start_idx < e1_end_idx):
                        # debug
                        num_of_exceptions += 1
                        #continue
                        
                        ## TODO: handle these cases for entity markers.
                        text_with_entity_marker = None
                        text_with_typed_entity_marker = None
                        e1_s_in_text_with_entity_marker = None
                        e1_e_in_text_with_entity_marker = None
                        e2_s_in_text_with_entity_marker = None
                        e2_e_in_text_with_entity_marker = None
                        e1_s_in_text_with_typed_entity_marker = None
                        e1_e_in_text_with_typed_entity_marker = None
                        e2_s_in_text_with_typed_entity_marker = None
                        e2_e_in_text_with_typed_entity_marker = None
                        
                    else:
                        # Add entity markers.
                        text_with_entity_marker = text
                        text_with_typed_entity_marker = text
                        
                        e1_typed_marker_s, e1_typed_marker_e = '[' + e1_type + ']', '[/' + e1_type + ']'
                        e2_typed_marker_s, e2_typed_marker_e = '[' + e2_type + ']', '[/' + e2_type + ']'
                        
                        # don't use replace since two entities are the same.
                        text_with_entity_marker = text_with_entity_marker[:e1_start_idx] + \
                                                  '[E1]' + e1_text + '[/E1]' + \
                                                  text_with_entity_marker[e1_end_idx:e2_start_idx] + \
                                                  '[E2]' + e2_text + '[/E2]' + \
                                                  text_with_entity_marker[e2_end_idx:]
                        
                        e1_s_in_text_with_entity_marker = text_with_entity_marker.index('[E1]') + len('[E1]')
                        e1_e_in_text_with_entity_marker = text_with_entity_marker.index('[/E1]')
                        e2_s_in_text_with_entity_marker = text_with_entity_marker.index('[E2]') + len('[E2]')
                        e2_e_in_text_with_entity_marker = text_with_entity_marker.index('[/E2]')
                        
                        text_with_typed_entity_marker = text_with_typed_entity_marker[:e1_start_idx] + \
                                                        e1_typed_marker_s + e1_text + e1_typed_marker_e + \
                                                        text_with_typed_entity_marker[e1_end_idx:e2_start_idx] + \
                                                        e2_typed_marker_s + e2_text + e2_typed_marker_e + \
                                                        text_with_typed_entity_marker[e2_end_idx:]
                                               
                        # don't use index() because two typed markers can be the same.
                        e1_s_in_text_with_typed_entity_marker = e1_start_idx + len(e1_typed_marker_s)
                        e1_e_in_text_with_typed_entity_marker = e1_end_idx + len(e1_typed_marker_s)
                        e2_s_in_text_with_typed_entity_marker = e2_start_idx + len(e1_typed_marker_s) + len(e1_typed_marker_e) + len(e2_typed_marker_s)
                        e2_e_in_text_with_typed_entity_marker = e2_end_idx + len(e1_typed_marker_s) + len(e1_typed_marker_e) + len(e2_typed_marker_s)
                            
                        # debug
                        if e1_text != text_with_entity_marker[e1_s_in_text_with_entity_marker:e1_e_in_text_with_entity_marker] or \
                           e1_text != text_with_typed_entity_marker[e1_s_in_text_with_typed_entity_marker:e1_e_in_text_with_typed_entity_marker] or \
                           e2_text != text_with_entity_marker[e2_s_in_text_with_entity_marker:e2_e_in_text_with_entity_marker] or \
                           e2_text != text_with_typed_entity_marker[e2_s_in_text_with_typed_entity_marker:e2_e_in_text_with_typed_entity_marker]:
                            print('text:', text)
                            print('e1_text:', e1_text, '/ e2_text:', e2_text)
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
                        
                    
                    relation = {'relation_type': relation_type, 
                                'relation_id': relation_types[relation_type]['id'],
                                'entity_1': e1_text,
                                'entity_1_idx': (e1_start_idx, e1_end_idx),
                                'entity_1_idx_in_text_with_entity_marker': (e1_s_in_text_with_entity_marker, e1_e_in_text_with_entity_marker),
                                'entity_1_idx_in_text_with_typed_entity_marker': (e1_s_in_text_with_typed_entity_marker, e1_e_in_text_with_typed_entity_marker),
                                'entity_1_type': e1_type,
                                'entity_1_type_id': 0,
                                'entity_2': e2_text,
                                'entity_2_idx': (e2_start_idx, e2_end_idx),
                                'entity_2_idx_in_text_with_entity_marker': (e2_s_in_text_with_entity_marker, e2_e_in_text_with_entity_marker),
                                'entity_2_idx_in_text_with_typed_entity_marker': (e2_s_in_text_with_typed_entity_marker, e2_e_in_text_with_typed_entity_marker),
                                'entity_2_type': e2_type, 
                                'entity_2_type_id': 0}
                    
                    # 'relation' item indicates the relation directionality. a.k.a symmetric or asymmetric relation.
                    # 'reverse' item is only used for undirected relations. 
                    # ('reverse' item) For testing phase, undirected samples are replicated, and the replicated samples are tagged as reverse. 
                    # ('reverse' item) So, if it's set to true, the model uses the second entity + the first entity instead of 
                    # ('reverse' item) the first entity + the second entity to classify both relation representation cases (A + B, B + A). 
                    sample = {"id": relation_type + '_' + str(sample_no) + '_' + str(pair_no),
                              "text": text,
                              "text_with_entity_marker": text_with_entity_marker,
                              "text_with_typed_entity_marker": text_with_typed_entity_marker,
                              "relation": [relation],
                              "directed": False,
                              "reverse": False,
                              }
                    
                    if doc_id in samples:
                        samples[doc_id].append(sample)
                    else:
                        samples[doc_id] = [sample]
                        
                    # debug
                    if relation_type in relation_cnt:
                        relation_cnt[relation_type] += 1
                    else:
                        relation_cnt[relation_type] = 1
    
    # debug
    for k, v in relation_cnt.items():
        print(k, v)
        
    return samples


def store_data(train, dev, test, fold_num, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    """
    ref: https://pythonhowtoprogram.com/how-to-write-multiple-json-objects-to-a-file-in-python-3/
    """
    train_txt = ''
    for sample in train:
        train_txt += json.dumps(sample)
        train_txt += '\n'
    
    outfile = os.path.join(output_dir, 'train_' + str(fold_num) + '.json')
    with open(outfile, "w") as f: 
        f.write(train_txt)
    
    
    """
    For development (validation) and test data, undirectional relation samples are replicated with reverse = True,
    so that the model classifies both relation representation cases (A + B, B + A). 
    """
    if dev is not None:
        dev_txt = ''
        for sample in dev:
            dev_txt += json.dumps(sample)
            dev_txt += '\n'
            
            '''
            if sample['directed'] is False:
                sample['reverse'] = True
                dev_txt += json.dumps(sample)
                dev_txt += '\n'
            '''
    
        outfile = os.path.join(output_dir, 'dev_' + str(fold_num) + '.json')
        with open(outfile, "w") as f: 
            f.write(dev_txt)
    
    test_txt = ''
    for sample in test:
        test_txt += json.dumps(sample)
        test_txt += '\n'
        
        '''
        if sample['directed'] is False:
            sample['reverse'] = True
            test_txt += json.dumps(sample)
            test_txt += '\n'
        '''

    outfile = os.path.join(output_dir, 'test_' + str(fold_num) + '.json')
    with open(outfile, "w") as f: 
        f.write(test_txt)


def split_and_save(doc_samples, output_dir, split_by_doc=False):
    doc_samples = {k: v for k, v in doc_samples.items() if len(v) > 0} # remove documents having no samples.
    
    if split_by_doc:
            
        # debugging
        '''
        unique_samples = []
        for doc, samples in doc_samples.items():
            for s in samples:
                if s in unique_samples:
                    print(s)
                    input('enter..')
                else:
                    unique_samples.append(s)
        '''

        keys = list(doc_samples.keys())
        random.shuffle(keys)
        
        total_num = 0

        #num_of_samples_for_eval = None if args.num_samples == -1 else args.num_samples
        num_of_samples_for_eval = None
        samples = []
        counter = 0
        
        for k in keys:
            #print(k, '/ num of samples:', len(doc_samples[k]))
            total_num += len(doc_samples[k])
            
            if num_of_samples_for_eval != None:
                counter += len(doc_samples[k])
                if counter > num_of_samples_for_eval:
                    max_idx = counter - num_of_samples_for_eval
                    samples.append(doc_samples[k][:max_idx])
                    break
                elif counter == num_of_samples_for_eval:
                    samples.append(doc_samples[k])
                    break
                else:
                    samples.append(doc_samples[k])
            else:
                samples.append(doc_samples[k])
                
        print('num of total samples:', total_num)

        # debugging
        '''
        print('len(unique_samples):', len(unique_samples))
        print('len(samples):', len(samples))
        
        samples_unique_samples = []
        for s in samples:
            for ss in s:
                if ss in samples_unique_samples:
                    print(ss)
                    input('enter..')
                else:
                    samples_unique_samples.append(ss)
        print('len(samples_unique_samples):', len(samples_unique_samples))			

        all_test_samples = [] 
        '''

        # debugging
        if num_of_samples_for_eval != None and num_of_samples_for_eval != len([item for sublist in samples for item in sublist]):
            input('sampling number is wrong!!')

        samples = np.array(samples)
        kfold = KFold(n_splits=10, shuffle=False)
        for fold_num, (train_index, test_index) in enumerate(kfold.split(samples)):
            train, test = samples[train_index], samples[test_index]
            #print("TRAIN len:", len(train_index), "TEST len:", len(test_index))
            #print("TRAIN:", train_index, "TEST:", test_index)
            #input('enter..')
            
            # no dev set in cv.
            store_doc_data(train, None, test, fold_num, output_dir)
            
    else:
        samples = []
        labels = []
        for v in doc_samples.values():
            for i in v:
                samples.append(i)
                labels.append(i['relation'][0]['relation_id'])
        
        samples = np.array(samples)
        labels = np.array(labels)
        
        kfold = StratifiedKFold(n_splits=10, shuffle=True)
        for fold_num, (train_index, test_index) in enumerate(kfold.split(samples, labels)):
        #kfold = KFold(n_splits=10, shuffle=True)
        #for fold_num, (train_index, test_index) in enumerate(kfold.split(samples)):
            train, test = samples[train_index], samples[test_index]
            #print("TRAIN len:", len(train_index), "TEST len:", len(test_index))
            #print("TRAIN:", train_index, "TEST:", test_index)
            #input('enter..')
            
            # no dev set in cv.
            store_data(train, None, test, fold_num, output_dir)
        
    logger.info("Finished and saved!")


def main():
    dataset_name = globals()['dataset_name']

    if dataset_name in ["P-putida"]:
        data_file = data_file_dict[dataset_name]
        data_file = os.path.join(current_working_dir, data_file)
        doc_samples = get_samples(data_file)
        #output_dir = data_file.rsplit('/', 1)[0] + '/10-fold-cv'
        output_dir = data_file.rsplit('/', 1)[0]
        split_and_save(doc_samples, output_dir)
    else:
        sys.exit('Unknown data!!')

main()	