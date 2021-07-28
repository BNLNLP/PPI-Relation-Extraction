#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 09:53:55 2019

@author: weetee

modified by: Gilchan Park
-- Find the modifications by the tag [GP].
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from .preprocessing_funcs import load_dataloaders
from .train_funcs import load_state, load_results, evaluate_, evaluate_results
from ..misc import save_as_pickle, load_pickle
# [GP][START] - error occurs. 11-27-2020
#import matplotlib.pyplot as plt
# [GP][END] - error occurs. 11-27-2020
import time
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

# [GP][START] - Set seed before initializing model for reproducibility. 03-22-2021
torch.manual_seed(0)
# [GP][END] - Set seed before initializing model for reproducibility. 03-22-2021


# [GP][START] - added dataset number parameter.
def train_and_fit(args, dataset_num):
# [GP][END] - added dataset number parameter.

	if args.fp16:
		from apex import amp
	else:
		amp = None
	
	cuda = torch.cuda.is_available()
	
	# [GP][START] - added CV set number for CV.
	# 			  - added dev set. 12-23-2020
	train_loader, dev_loader, test_loader, train_len, dev_len, test_len = load_dataloaders(args, dataset_num)
	# [GP][END] - added CV set number for CV.
	logger.info("Loaded %d Training samples." % train_len)
	
	if args.model_no == 0:
		from ..model.BERT.modeling_bert import BertModel as Model
		model = args.model_size #'bert-base-uncased'
		lower_case = True
		model_name = 'BERT'
		net = Model.from_pretrained(model, force_download=False, \
								model_size=args.model_size,
								task='classification' if args.task != 'fewrel' else 'fewrel',\
								n_classes_=args.num_classes)
	elif args.model_no == 1:
		from ..model.ALBERT.modeling_albert import AlbertModel as Model
		model = args.model_size #'albert-base-v2'
		lower_case = True
		model_name = 'ALBERT'
		net = Model.from_pretrained(model, force_download=False, \
								model_size=args.model_size,
								task='classification' if args.task != 'fewrel' else 'fewrel',\
								n_classes_=args.num_classes)
	elif args.model_no == 2: # BioBert
		from ..model.BERT.modeling_bert import BertModel, BertConfig
		model = 'bert-base-uncased'
		lower_case = False
		model_name = 'BioBERT'
		config = BertConfig.from_pretrained('./additional_models/biobert_v1.1_pubmed/bert_config.json')
		net = BertModel.from_pretrained(pretrained_model_name_or_path='./additional_models/biobert_v1.1_pubmed/biobert_v1.1_pubmed.bin', 
										  config=config,
										  force_download=False, \
										  model_size='bert-base-uncased',
										  task='classification' if args.task != 'fewrel' else 'fewrel',\
										  n_classes_=args.num_classes)

	tokenizer = load_pickle("%s_tokenizer.pkl" % model_name)
	net.resize_token_embeddings(len(tokenizer))
	e1_id = tokenizer.convert_tokens_to_ids('[E1]')
	e2_id = tokenizer.convert_tokens_to_ids('[E2]')
	assert e1_id != e2_id != 1
	
	if cuda:
		net.cuda()
		
	logger.info("FREEZING MOST HIDDEN LAYERS...")
	if args.model_no == 0:
		unfrozen_layers = ["classifier", "pooler", "encoder.layer.11", \
						   "classification_layer", "blanks_linear", "lm_linear", "cls"]
	elif args.model_no == 1:
		unfrozen_layers = ["classifier", "pooler", "classification_layer",\
						   "blanks_linear", "lm_linear", "cls",\
						   "albert_layer_groups.0.albert_layers.0.ffn"]
	elif args.model_no == 2:
		unfrozen_layers = ["classifier", "pooler", "encoder.layer.11", \
						   "classification_layer", "blanks_linear", "lm_linear", "cls"]
		
	for name, param in net.named_parameters():
		if not any([layer in name for layer in unfrozen_layers]):
			print("[FROZE]: %s" % name)
			param.requires_grad = False
		else:
			print("[FREE]: %s" % name)
			param.requires_grad = True
	
	if args.use_pretrained_blanks == 1:
		logger.info("Loading model pre-trained on blanks at ./data/test_checkpoint_%d.pth.tar..." % args.model_no)
		checkpoint_path = "./data/test_checkpoint_%d.pth.tar" % args.model_no
		checkpoint = torch.load(checkpoint_path)
		model_dict = net.state_dict()
		pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict.keys()}
		model_dict.update(pretrained_dict)
		net.load_state_dict(pretrained_dict, strict=False)
		del checkpoint, pretrained_dict, model_dict
	
	# [GP][START] - Binary Crosss Entropy (BCE) for one-class classification. 01-06-2021
	if args.do_one_class_classification:
		#criterion = nn.BCELoss()
		criterion = nn.BCEWithLogitsLoss()
	# [GP][END] - Binary Crosss Entropy (BCE) for one-class classification. 01-06-2021
	else:
		criterion = nn.CrossEntropyLoss(ignore_index=-1)

	optimizer = optim.Adam([{"params":net.parameters(), "lr": args.lr}])
	
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4,6,8,12,15,18,20,22,\
																	  24,26,30], gamma=0.8)
	
	start_epoch, best_pred, amp_checkpoint = load_state(net, optimizer, scheduler, args, load_best=False)  
	
	if (args.fp16) and (amp is not None):
		logger.info("Using fp16...")
		net, optimizer = amp.initialize(net, optimizer, opt_level='O2')
		if amp_checkpoint is not None:
			amp.load_state_dict(amp_checkpoint)
		scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4,6,8,12,15,18,20,22,\
																		  24,26,30], gamma=0.8)
	
	losses_per_epoch, accuracy_per_epoch, test_f1_per_epoch = load_results(args.model_no)
	
	logger.info("Starting training process...")
	pad_id = tokenizer.pad_token_id
	mask_id = tokenizer.mask_token_id
	update_size = len(train_loader)//10
	
	# debugging
	print('update_size:', update_size)
	print('len(train_loader):', len(train_loader))
	
	# [GP][START] - save result per epoch.
	test_result_per_epoch = []
	# [GP][END] - save result per epoch.

	for epoch in range(start_epoch, args.num_epochs):
		start_time = time.time()
		net.train(); total_loss = 0.0; losses_per_batch = []; total_acc = 0.0; accuracy_per_batch = []
		for i, data in enumerate(train_loader, 0):
			x, e1_e2_start, labels, _,_,_ = data
			attention_mask = (x != pad_id).float()
			token_type_ids = torch.zeros((x.shape[0], x.shape[1])).long()
			
			
			# [GP]
			print(net)
			print(type(data))
			print('data:', data)
			print('x:', x)
			print('e1_e2_start:', e1_e2_start)
			print('labels:', labels)
			print('attention_mask:', attention_mask)
			print('token_type_ids:', token_type_ids)
			
			#for i in x:
			#	print(tokenizer.decode(i))
			#	print(tokenizer.convert_ids_to_tokens(i))
			#input('enter.')
			
			
			if cuda:
				x = x.cuda()
				labels = labels.cuda()
				attention_mask = attention_mask.cuda()
				token_type_ids = token_type_ids.cuda()
				
			classification_logits = net(x, token_type_ids=token_type_ids, attention_mask=attention_mask, Q=None,\
										e1_e2_start=e1_e2_start)
			
			
			print('classification_logits.shape:', classification_logits.shape)
			print('classification_logits:', classification_logits)
			print('labels:', labels)
			print('labels.squeeze(1).shape:', labels.squeeze(1).shape)
			print('labels.squeeze(1):', labels.squeeze(1))
				
			input('enter.')
			
			
			
			#return classification_logits, labels, net, tokenizer # for debugging now
			
			# [GP][START] - Binary Crosss Entropy (BCE) for one-class classification. 01-06-2021
			if args.do_one_class_classification:
				targets = torch.ones(len(labels.squeeze(1)), 1)
				if cuda:
					targets = targets.cuda()
			
				#loss = criterion(torch.sigmoid(classification_logits), targets)	# use sigmoid in case of nn.BCELoss()
				loss = criterion(classification_logits, targets)
				
				# debugging
				'''
				print('classification_logits:', classification_logits)
				print('labels:', labels)
				print('labels.squeeze(1):', labels.squeeze(1))
				'''	
			# [GP][END] - Binary Crosss Entropy (BCE) for one-class classification. 01-06-2021
			else:
				loss = criterion(classification_logits, labels.squeeze(1))

			loss = loss/args.gradient_acc_steps

			if args.fp16:
				with amp.scale_loss(loss, optimizer) as scaled_loss:
					scaled_loss.backward()
			else:
				loss.backward()
			
			if args.fp16:
				grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
			else:
				grad_norm = clip_grad_norm_(net.parameters(), args.max_norm)
			
			if (i % args.gradient_acc_steps) == 0:
				optimizer.step()
				optimizer.zero_grad()
			
			total_loss += loss.item()

			# [GP][START] - added one-class classification parameters. 01-06-2021
			total_acc += evaluate_(classification_logits, labels, cuda, ignore_idx=-1, \
								   do_one_cls_classification=args.do_one_class_classification, \
								   threshold=args.threshold)[0]
			# [GP][END] - added one-class classification parameters. 01-06-2021

			# debugging
			'''
			print('loss:', loss)
			print('loss.item():', loss.item())
			print('total_loss:', total_loss)
			print('total_acc:', total_acc)
			input('enter...')
			'''
			
			if (i % update_size) == (update_size - 1):
				losses_per_batch.append(args.gradient_acc_steps*total_loss/update_size)
				accuracy_per_batch.append(total_acc/update_size)
				print('[Epoch: %d, %5d/ %d points] total loss, accuracy per batch: %.3f, %.3f' %
					  (epoch + 1, (i + 1)*args.batch_size, train_len, losses_per_batch[-1], accuracy_per_batch[-1]))
				total_loss = 0.0; total_acc = 0.0

		scheduler.step()
		
		# [GP][START] - use dev set during training. 12-23-2020, added one-class classification parameters. 01-06-2021
		if dev_loader is not None:
			results = evaluate_results(net, dev_loader, pad_id, cuda, \
									   do_one_cls_classification=args.do_one_class_classification, \
									   threshold=args.threshold)
		else:
			#results = evaluate_results(net, test_loader, pad_id, cuda)
			results = evaluate_results(net, test_loader, pad_id, cuda, \
									   do_one_cls_classification=args.do_one_class_classification, \
									   threshold=args.threshold)
		# [GP][END] - use dev set during training. 12-23-2020, added one-class classification parameters. 01-06-2021
		losses_per_epoch.append(sum(losses_per_batch)/len(losses_per_batch))
		accuracy_per_epoch.append(sum(accuracy_per_batch)/len(accuracy_per_batch))
		test_f1_per_epoch.append(results['f1'])
		# [GP][START] - save result per epoch.
		test_result_per_epoch.append(results)
		# [GP][END] - save result per epoch.
		
		print("Epoch finished, took %.2f seconds." % (time.time() - start_time))
		print("Losses at Epoch %d: %.7f" % (epoch + 1, losses_per_epoch[-1]))
		print("Train accuracy at Epoch %d: %.7f" % (epoch + 1, accuracy_per_epoch[-1]))
		print("Test f1 at Epoch %d: %.7f" % (epoch + 1, test_f1_per_epoch[-1]))
		
		if accuracy_per_epoch[-1] > best_pred:
			best_pred = accuracy_per_epoch[-1]
			torch.save({
					'epoch': epoch + 1,\
					'state_dict': net.state_dict(),\
					'best_acc': accuracy_per_epoch[-1],\
					'optimizer' : optimizer.state_dict(),\
					'scheduler' : scheduler.state_dict(),\
					'amp': amp.state_dict() if amp is not None else amp
				}, os.path.join("./data/" , "task_test_model_best_%d.pth.tar" % args.model_no))
		
		if (epoch % 1) == 0:
			save_as_pickle("task_test_losses_per_epoch_%d.pkl" % args.model_no, losses_per_epoch)
			save_as_pickle("task_train_accuracy_per_epoch_%d.pkl" % args.model_no, accuracy_per_epoch)
			save_as_pickle("task_test_f1_per_epoch_%d.pkl" % args.model_no, test_f1_per_epoch)
			torch.save({
					'epoch': epoch + 1,\
					'state_dict': net.state_dict(),\
					'best_acc': accuracy_per_epoch[-1],\
					'optimizer' : optimizer.state_dict(),\
					'scheduler' : scheduler.state_dict(),\
					'amp': amp.state_dict() if amp is not None else amp
				}, os.path.join("./data/" , "task_test_checkpoint_%d.pth.tar" % args.model_no))
	
	logger.info("Finished Training!")
	
	
	# [GP][START] - for train/validation(dev)/test, evaluate test data. 12-23-2020
	#			  - added one-class classification parameters. 01-06-2021
	for r in test_result_per_epoch:
		print('result:', r)
	
	print('--------------------------------------------------------')

	test_result = evaluate_results(net, test_loader, pad_id, cuda, \
								   do_one_cls_classification=args.do_one_class_classification, \
								   threshold=args.threshold)
	
	print('result:', test_result)
	# [GP][START] - for train/validation(dev)/test, evaluate test data. 12-23-2020
	
	# [GP][START] - error occurs. 11-27-2020
	'''
	fig = plt.figure(figsize=(20,20))
	ax = fig.add_subplot(111)
	ax.scatter([e for e in range(len(losses_per_epoch))], losses_per_epoch)
	ax.tick_params(axis="both", length=2, width=1, labelsize=14)
	ax.set_xlabel("Epoch", fontsize=22)
	ax.set_ylabel("Training Loss per batch", fontsize=22)
	ax.set_title("Training Loss vs Epoch", fontsize=32)
	plt.savefig(os.path.join("./data/" ,"task_loss_vs_epoch_%d.png" % args.model_no))
	
	fig2 = plt.figure(figsize=(20,20))
	ax2 = fig2.add_subplot(111)
	ax2.scatter([e for e in range(len(accuracy_per_epoch))], accuracy_per_epoch)
	ax2.tick_params(axis="both", length=2, width=1, labelsize=14)
	ax2.set_xlabel("Epoch", fontsize=22)
	ax2.set_ylabel("Training Accuracy", fontsize=22)
	ax2.set_title("Training Accuracy vs Epoch", fontsize=32)
	plt.savefig(os.path.join("./data/" ,"task_train_accuracy_vs_epoch_%d.png" % args.model_no))
	
	fig3 = plt.figure(figsize=(20,20))
	ax3 = fig3.add_subplot(111)
	ax3.scatter([e for e in range(len(test_f1_per_epoch))], test_f1_per_epoch)
	ax3.tick_params(axis="both", length=2, width=1, labelsize=14)
	ax3.set_xlabel("Epoch", fontsize=22)
	ax3.set_ylabel("Test F1 Accuracy", fontsize=22)
	ax3.set_title("Test F1 vs Epoch", fontsize=32)
	plt.savefig(os.path.join("./data/" ,"task_test_f1_vs_epoch_%d.png" % args.model_no))
	'''
	# [GP][END] - error occurs. 11-27-2020
	
	#return net, test_result_per_epoch
	return net, test_result