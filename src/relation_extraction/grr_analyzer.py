import os
import csv
import re

from sklearn.metrics import f1_score, accuracy_score, classification_report, recall_score, precision_score, precision_recall_fscore_support


#output_file = "/hpcgpfs01/scratch/gpark/RE_results/GRR/P-putida/dmis-lab/biobert-base-cased-v1.1/STANDARD_mention_pooling_ac/predict_outputs.txt"
output_file = "/hpcgpfs01/scratch/gpark/RE_results/GRR/P-species/dmis-lab/biobert-base-cased-v1.1/STANDARD_mention_pooling_ac/predict_outputs.txt"

pred, true = [], []
pairs = {}
texts = set()

f = open(output_file, "r")
for line in f.readlines()[1:]:
    data = line.split('\t')
    
    p = data[4].strip()
    t = data[5].strip()
    
    pred.append(p)
    true.append(t)
    
    e1 = data[1].lower().strip()
    e2 = data[2].lower().strip()
    
    '''
    if e1.startswith('pp') or e2.startswith('pp'):
        print(e1)
        print(e2)
        input('enter..')
    '''
    e1 = re.sub(r'[^a-zA-Z0-9]', '', e1)
    e2 = re.sub(r'[^a-zA-Z0-9]', '', e2)
    
    if e1.startswith('copz') or e1.startswith('copa'):
        #print(e1)
        e1 = re.sub(r'[0-9]', '', e1)
        #print(e1 + '-' + e2)
        #input('enter..')
    
    if e2.startswith('copz') or e2.startswith('copa'):
        #print(e2)
        e2 = re.sub(r'[0-9]', '', e2)
        #print(e2 + '-' + e1)
        #input('enter..')
        
    if e1 == 'iscrsua':
        e1 = ['iscr', 'iscs', 'iscu', 'isca']
    else:
        e1 = [e1]
    
    if e2 == 'iscrsua':
        e2 = ['iscr', 'iscs', 'iscu', 'isca']
    else:
        e2 = [e2]
    
    for x in e1:
        for y in e2:
            k = x + '-' + y
            if k in pairs:
                pairs[k].append(p)
            else:
                pairs[k] = [p]
                
            k = y + '-' + x
            if k in pairs:
                pairs[k].append(p)
            else:
                pairs[k] = [p]

    texts.add(data[3])

#f1score = f1_score(y_pred=pred, y_true=true, average='micro', labels=label_list)
precision, recall, f1, _ = precision_recall_fscore_support(y_pred=pred, y_true=true, \
                                                           average='micro')
accuracy = accuracy_score(true, pred) # TODO: ignore 'DDI-false' for DDI evaluation.

print("<sklearn> - classification_report")
#print(classification_report(true, pred, digits=4, labels=label_list))
print(classification_report(true, pred, digits=4))
print('<sklearn> precision:', precision)
print('<sklearn> recall:', recall)
print('<sklearn> f1:', f1) # this is the same as f1_score.
print('<sklearn> accuracy:', accuracy)


#reg_precise_file = "/direct/sdcc+u/gpark/BER-NLP/RE/datasets/GRR/P-putida/Pputida.regulations.fixed.tsv"
reg_precise_file = "/direct/sdcc+u/gpark/BER-NLP/PPI-Relation-Extraction/datasets/GRR/P-species/Pseudomonas_genus.largest_species.regulations.tsv"

f = open(reg_precise_file, "r")
tsv_reader = csv.reader(f, delimiter="\t")

# skip the header
next(tsv_reader)

rp_pairs = []

output_txt = ''
for row in tsv_reader:
    regulator = row[1]
    regulator_name = row[2]
    regulatee = row[3]
    regulatee_name = row[4]
    
    regulator = regulator.lower().strip()
    regulator_name = regulator_name.lower().strip()
    regulatee = regulatee.lower().strip()
    regulatee_name = regulatee_name.lower().strip()
    
    regulator = re.sub(r'[^a-zA-Z0-9]', '', regulator)
    regulator_name = re.sub(r'[^a-zA-Z0-9]', '', regulator_name)
    regulatee = re.sub(r'[^a-zA-Z0-9]', '', regulatee)
    regulatee_name = re.sub(r'[^a-zA-Z0-9]', '', regulatee_name)

    rp_pairs.append(regulator + '-' + regulatee)
    rp_pairs.append(regulatee + '-' + regulator)
    rp_pairs.append(regulator_name + '-' + regulatee_name)
    rp_pairs.append(regulatee_name + '-' + regulator_name)


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

#print(pairs)
#print(rp_pairs)
print(intersection(list(pairs.keys()), rp_pairs))
print(len(intersection(list(pairs.keys()), rp_pairs)))

dup_checker = []
rp_cnt = 0
total_cnt = 0
for no, i in enumerate(intersection(list(pairs.keys()), rp_pairs)):
    if i not in dup_checker:
        rp_cnt += 1
        print(rp_cnt, i, len(pairs[i]), pairs[i])
        total_cnt += len(pairs[i])
        #input('enter..')
        
    dup_checker.append(i)
    dup_checker.append(i.split('-')[1] + '-' + i.split('-')[0])
    
print(total_cnt) 
    


'''
#fig_text = [x for x in texts if 'fig.' in x.lower() or 'figure' in x.lower()]
fig_text = [x for x in texts if 'fig' in x.lower()]
print(len(fig_text))

for t in fig_text:
    print(t)
'''

    
    
    
    
    
    
    
    
    
    
    
    
    
    