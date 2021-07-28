#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 09:37:26 2019

@author: weetee

modified by: Gilchan Park
-- Find the modifications by the tag [GP].
"""
import os
import math
import torch
import torch.nn as nn
from ..misc import save_as_pickle, load_pickle
from seqeval.metrics import precision_score, recall_score, f1_score
import logging
from tqdm import tqdm


logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

def load_state(net, optimizer, scheduler, args, load_best=False):
    """ Loads saved model and optimizer states if exists """
    base_path = "./data/"
    amp_checkpoint = None
    checkpoint_path = os.path.join(base_path,"task_test_checkpoint_%d.pth.tar" % args.model_no)
    best_path = os.path.join(base_path,"task_test_model_best_%d.pth.tar" % args.model_no)
    start_epoch, best_pred, checkpoint = 0, 0, None
    if (load_best == True) and os.path.isfile(best_path):
        checkpoint = torch.load(best_path)
        logger.info("Loaded best model.")
    elif os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        logger.info("Loaded checkpoint model.")
    if checkpoint != None:
        start_epoch = checkpoint['epoch']
        best_pred = checkpoint['best_acc']
        net.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        amp_checkpoint = checkpoint['amp']
        logger.info("Loaded model and optimizer.")    
    return start_epoch, best_pred, amp_checkpoint

def load_results(model_no=0):
    """ Loads saved results if exists """
    losses_path = "./data/task_test_losses_per_epoch_%d.pkl" % model_no
    accuracy_path = "./data/task_train_accuracy_per_epoch_%d.pkl" % model_no
    f1_path = "./data/task_test_f1_per_epoch_%d.pkl" % model_no
    if os.path.isfile(losses_path) and os.path.isfile(accuracy_path) and os.path.isfile(f1_path):
        losses_per_epoch = load_pickle("task_test_losses_per_epoch_%d.pkl" % model_no)
        accuracy_per_epoch = load_pickle("task_train_accuracy_per_epoch_%d.pkl" % model_no)
        f1_per_epoch = load_pickle("task_test_f1_per_epoch_%d.pkl" % model_no)
        logger.info("Loaded results buffer")
    else:
        losses_per_epoch, accuracy_per_epoch, f1_per_epoch = [], [], []
    return losses_per_epoch, accuracy_per_epoch, f1_per_epoch


# [GP][START] - added one-class classification parameters. 01-06-2021
def evaluate_(output, labels, cuda, ignore_idx, do_one_cls_classification=False, threshold=None):
# [GP][END] - added one-class classification parameters. 01-06-2021
	### ignore index 0 (padding) when calculating accuracy
	idxs = (labels != ignore_idx).squeeze()
	
	# [GP][START] - use sigmoid for one-class classification. 01-06-2021
	if do_one_cls_classification:

		sig_vals = [x for sublist in torch.sigmoid(output).tolist() for x in sublist]
		o_labels = []
		for val in sig_vals:
			if val > threshold:
				o_labels.append(0)
			else:
				o_labels.append(1)
		o_labels = torch.IntTensor(o_labels)
		if cuda:
			o_labels = o_labels.cuda()
	# [GP][END] - use sigmoid for one-class classification. 01-06-2021
	else:
		o_labels = torch.softmax(output, dim=1).max(1)[1]

	l = labels.squeeze()[idxs]; o = o_labels[idxs]

	# [GP][START] - check tensor size for TypeError: len() of a 0-d tensor.
	if bool(idxs.size()) and len(idxs) > 1:
	# [GP][END] - check tensor size for TypeError: len() of a 0-d tensor.
		acc = (l == o).sum().item()/len(idxs)
	else:
		acc = (l == o).sum().item()
	l = l.cpu().numpy().tolist() if l.is_cuda else l.numpy().tolist()
	o = o.cpu().numpy().tolist() if o.is_cuda else o.numpy().tolist()

	return acc, (o, l)

# [GP][START] - added one-class classification parameters. 01-06-2021
def evaluate_results(net, test_loader, pad_id, cuda, do_one_cls_classification=False, threshold=None):
# [GP][END] - added one-class classification parameters. 01-06-2021
	logger.info("Evaluating test samples...")
	acc = 0; out_labels = []; true_labels = []
	net.eval()
	with torch.no_grad():
		for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
			x, e1_e2_start, labels, _,_,_ = data
			attention_mask = (x != pad_id).float()
			token_type_ids = torch.zeros((x.shape[0], x.shape[1])).long()

			if cuda:
				x = x.cuda()
				labels = labels.cuda()
				attention_mask = attention_mask.cuda()
				token_type_ids = token_type_ids.cuda()

			classification_logits = net(x, token_type_ids=token_type_ids, attention_mask=attention_mask, Q=None,\
										e1_e2_start=e1_e2_start)

			# [GP][START] - use sigmoid for one-class classification. 01-06-2021
			accuracy, (o, l) = evaluate_(classification_logits, labels, cuda, ignore_idx=-1, \
										 do_one_cls_classification=do_one_cls_classification, \
										 threshold=threshold)
			# [GP][END] - added one-class classification parameters. 01-06-2021
			out_labels.append([str(i) for i in o]); true_labels.append([str(i) for i in l])
			acc += accuracy
			
			'''
			print('accuracy:', accuracy)
			print('acc:', acc)
			print('o:', o)
			print('l:', l)
			print('out_labels:', out_labels)
			print('true_labels:', true_labels)
			
			#input('enter...')
			'''	

	accuracy = acc/(i + 1)
	
	# [GP][START] - use sklearn metrics for evaluation.
	'''
	results = {
		"accuracy": accuracy,
		"precision": precision_score(true_labels, out_labels),
		"recall": recall_score(true_labels, out_labels),
		"f1": f1_score(true_labels, out_labels)
	}
	'''
	
	from sklearn.metrics import precision_recall_fscore_support
	import numpy as np
	
	'''
	print('true_labels:', true_labels)
	print('out_labels:', out_labels)
	print('accuracy:', accuracy)
	print('precision:', precision_score(true_labels_one_d, out_labels_one_d))
	print('recall:', recall_score(true_labels_one_d, out_labels_one_d))
	print('f1_score:', f1_score(true_labels_one_d, out_labels_one_d))
	'''
	true_labels_flattened = np.concatenate(true_labels)
	out_labels_flattened = np.concatenate(out_labels)
	#true_labels_flattened = list(true_labels_flattened)
	#out_labels_flattened = list(out_labels_flattened)

	precision, recall, f1, _ = precision_recall_fscore_support(true_labels_flattened, out_labels_flattened, average='weighted')
	results = {
		"accuracy": accuracy,
		"precision": precision,
		"recall": recall,
		"f1": f1
	}
	# [GP][END] - use sklearn metrics for evaluation.
	
	
	logger.info("***** Eval results *****")
	for key in sorted(results.keys()):
		logger.info("  %s = %s", key, str(results[key]))

	return results
	