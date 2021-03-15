#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 17:40:16 2019

@author: weetee

modified by: Gilchan Park
-- Find the modifications by the tag [GP].
"""

from src.tasks.trainer import train_and_fit
from src.tasks.infer import infer_from_trained, FewRel
import logging
from argparse import ArgumentParser
import os
import numpy as np
import csv

'''
This fine-tunes the BERT model on SemEval, FewRel tasks
'''

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

if __name__ == "__main__":
	parser = ArgumentParser()
	# [GP][START] - BioCreative added. 11-26-2020
	#			  - AImed, BioInfer, HPRD50, IEPA, LLL added. 02-19-2021
	parser.add_argument("--task", type=str, default='semeval', help='semeval, fewrel, BioCreative, BioCreative_BNL, AImed, BioInfer, HPRD50, IEPA, LLL, AImed_BioInfer_HPRD50_IEPA_LLL')
	# [GP][END] - BioCreative added. 11-26-2020
	parser.add_argument("--train_data", type=str, default='./data/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT', \
						help="training data .txt file path")
	parser.add_argument("--test_data", type=str, default='./data/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT', \
						help="test data .txt file path")
	parser.add_argument("--use_pretrained_blanks", type=int, default=0, help="0: Don't use pre-trained blanks model, 1: use pre-trained blanks model")
	parser.add_argument("--num_classes", type=int, default=19, help='number of relation classes')
	parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
	parser.add_argument("--gradient_acc_steps", type=int, default=2, help="No. of steps of gradient accumulation")
	parser.add_argument("--max_norm", type=float, default=1.0, help="Clipped gradient norm")
	parser.add_argument("--fp16", type=int, default=0, help="1: use mixed precision ; 0: use floating point 32") # mixed precision doesn't seem to train well
	parser.add_argument("--num_epochs", type=int, default=11, help="No of epochs")
	parser.add_argument("--lr", type=float, default=0.00007, help="learning rate")
	parser.add_argument("--model_no", type=int, default=0, help='''Model ID: 0 - BERT\n
																			 1 - ALBERT\n
																			 2 - BioBERT''')
	parser.add_argument("--model_size", type=str, default='bert-base-uncased', help="For BERT: 'bert-base-uncased', \
																								'bert-large-uncased',\
																					For ALBERT: 'albert-base-v2',\
																								'albert-large-v2'\
																					For BioBERT: 'bert-base-uncased' (biobert_v1.1_pubmed)")
	parser.add_argument("--train", type=int, default=1, help="0: Don't train, 1: train")
	parser.add_argument("--infer", type=int, default=1, help="0: Don't infer, 1: Infer")
	
	# [GP][START] - arguments for cross-validation. 11-29-2020
	parser.add_argument("--num_samples", type=int, default=-1, help="No of samples for evaluation. -1 means to use all samples.")
	parser.add_argument("--do_cross_validation", action="store_true", help="Whether to use cross-validation for evaluation.")
	parser.add_argument("--num_of_folds", default=10, type=int, help="The number of folds for the cross validation.")
	parser.add_argument("--ratio", type=str, default='60-20-20', help="train/dev/test ratio: 80-10-10, 70-15-15, 60-20-20")
	parser.add_argument("--eval_value", type=str, help="best: get the best value among epochs, last: get the value of last epoch")
	parser.add_argument("--do_one_class_classification", action="store_true", help="Whether to use sigmoid for outputs rather than softmax. \
																					this is used for testing BioCreative 'PPIm' label.")
	parser.add_argument("--threshold", default=0.5, type=float, help="threshold value to determine one-class outputs. \
																	  this is used together with do_one_class_classification.")																	
	parser.add_argument("--predefined_cls", type=int, help="predefined labels such as PPIm.")
	parser.add_argument("--result_dir", type=str, help="result directory path")
	# [GP][END] - arguments for cross-validation. 11-29-2020
	
	# [GP][START] - added relation mapper path for infer task. 02-27-2021
	parser.add_argument("--rm_file", type=str, help="(for infer) relation mapper path")	
	# [GP][END] - added relation mapper path for infer task. 02-27-2021
	
	args = parser.parse_args()
	
	if (args.train == 1) and (args.task != 'fewrel'):
		# [GP][START] - set the number of datasets for training and evaluation, and save CV results. 11-29-2020
		num_of_datasets = 1 # if not CV, there is only one dataset.
		if args.do_cross_validation:
			num_of_datasets = args.num_of_folds
		
		if args.num_samples == -1:
			if not os.path.exists(args.result_dir + '/all'):
				os.makedirs(args.result_dir + '/all')
		else:
			if not os.path.exists(args.result_dir + '/' + str(args.num_samples)):
				os.makedirs(args.result_dir + '/' + str(args.num_samples))
			
		if args.model_no == 0:
			result_file = 'BERT_'
		elif args.model_no == 1:
			result_file = 'ALBERT_'
		elif args.model_no == 2:
			result_file = 'BioBERT_'
		
		if args.use_pretrained_blanks:
			result_file += 'with_MTB_cv_result.txt'
		else:
			result_file += 'without_MTB_cv_result.txt'
		
		if args.num_samples == -1:
			result_file = os.path.join(args.result_dir + '/all', result_file)
		else:
			result_file = os.path.join(args.result_dir + '/' + str(args.num_samples), result_file)
		
		if args.eval_value == 'best':
			best_f1_per_cv_set = []
			best_f1_precision_per_cv_set = []
			best_f1_recall_per_cv_set = []
			best_f1_accuracy_per_cv_set = []
		elif args.eval_value == 'last':
			last_f1_per_cv_set = []
			last_f1_precision_per_cv_set = []
			last_f1_recall_per_cv_set = []
			last_f1_accuracy_per_cv_set = []
			
		for dataset_num in range(num_of_datasets):
			
			net, result = train_and_fit(args, dataset_num)
			
			last_f1 = 0
			last_f1_precision = 0
			last_f1_recall = 0
			last_f1_accuracy = 0 # accuracy of the last epoch.
			
			with open(result_file, 'a') as fp:
				out_s = 'cv ' if args.do_cross_validation else ''
				out_s += "set: {set:d} / accuracy: {accuracy:.2f} / precision: {precision:.2f} / recall: {recall:.2f} / f1: {f1:.2f}\n".format(set=dataset_num, accuracy=result['accuracy'], precision=result['precision'], recall=result['recall'], f1=result['f1'])
				fp.write(out_s + '--------------------------\n')
				
				print(out_s)

			last_f1_per_cv_set.append(result['f1'])
			last_f1_precision_per_cv_set.append(result['precision'])
			last_f1_recall_per_cv_set.append(result['recall'])
			last_f1_accuracy_per_cv_set.append(result['accuracy'])

			"""
			net, test_result_per_epoch = train_and_fit(args, dataset_num)
			
			if args.eval_value == 'best':
				best_f1 = 0
				best_f1_precision = 0
				best_f1_recall = 0
				best_f1_accuracy = 0 # accuracy when F1 is the best.
				with open(result_file, 'a') as fp:
					for epoch, result in enumerate(test_result_per_epoch, 1):
						fp.write("cv set: %d / epoch: %d / accuracy: %.2f / precision: %.2f / recall: %.2f / f1: %.2f\n" % (dataset_num, epoch, result['accuracy'], result['precision'], result['recall'], result['f1']))
						print("cv set: %d / epoch: %d / accuracy: %.2f / precision: %.2f / recall: %.2f / f1: %.2f\n" % (dataset_num, epoch, result['accuracy'], result['precision'], result['recall'], result['f1']))
						if result['f1'] > best_f1:
							best_f1 = result['f1']
							best_f1_precision = result['precision']
							best_f1_recall = result['recall']
							best_f1_accuracy = result['accuracy']
							
					fp.write('--------------------------\n')
				
				best_f1_per_cv_set.append(best_f1)
				best_f1_precision_per_cv_set.append(best_f1_precision)
				best_f1_recall_per_cv_set.append(best_f1_recall)
				best_f1_accuracy_per_cv_set.append(best_f1_accuracy)
			
			elif args.eval_value == 'last':
				last_f1 = 0
				last_f1_precision = 0
				last_f1_recall = 0
				last_f1_accuracy = 0 # accuracy of the last epoch.
				
				with open(result_file, 'a') as fp:
					for epoch, result in enumerate(test_result_per_epoch, 1):
						fp.write("cv set: %d / epoch: %d / accuracy: %.2f / precision: %.2f / recall: %.2f / f1: %.2f\n" % (dataset_num, epoch, result['accuracy'], result['precision'], result['recall'], result['f1']))
						print("cv set: %d / epoch: %d / accuracy: %.2f / precision: %.2f / recall: %.2f / f1: %.2f\n" % (dataset_num, epoch, result['accuracy'], result['precision'], result['recall'], result['f1']))
							
					fp.write('--------------------------\n')
				
				last_f1_per_cv_set.append(test_result_per_epoch[-1]['f1'])
				last_f1_precision_per_cv_set.append(test_result_per_epoch[-1]['precision'])
				last_f1_recall_per_cv_set.append(test_result_per_epoch[-1]['recall'])
				last_f1_accuracy_per_cv_set.append(test_result_per_epoch[-1]['accuracy'])
			"""	
			
			if args.do_cross_validation:
				# remove files for the next CV set.
				os.remove(os.path.join("./data/" , "task_test_model_best_%d.pth.tar" % args.model_no))
				os.remove(os.path.join("./data/" , "task_test_checkpoint_%d.pth.tar" % args.model_no))
				os.remove(os.path.join("./data/" , "task_test_losses_per_epoch_%d.pkl" % args.model_no))
				os.remove(os.path.join("./data/" , "task_train_accuracy_per_epoch_%d.pkl" % args.model_no))
				os.remove(os.path.join("./data/" , "task_test_f1_per_epoch_%d.pkl" % args.model_no))

		if args.eval_value == 'best':
			best_f1_accuracy_per_cv_set = np.array(best_f1_accuracy_per_cv_set)
			best_f1_precision_per_cv_set = np.array(best_f1_precision_per_cv_set)
			best_f1_recall_per_cv_set = np.array(best_f1_recall_per_cv_set)
			best_f1_per_cv_set = np.array(best_f1_per_cv_set)
			
			print(">> Average Accuracy: %.2f with a std of %.2f\n" % (best_f1_accuracy_per_cv_set.mean(), best_f1_accuracy_per_cv_set.std()))
			print(">> Average Precision: %.2f with a std of %.2f\n" % (best_f1_precision_per_cv_set.mean(), best_f1_precision_per_cv_set.std()))
			print(">> Average Recall: %.2f with a std of %.2f\n" % (best_f1_recall_per_cv_set.mean(), best_f1_recall_per_cv_set.std()))
			print(">> Average F1: %.2f with a std of %.2f\n" % (best_f1_per_cv_set.mean(), best_f1_per_cv_set.std()))
			
			with open(result_file, 'a') as fp:
				fp.write(">> Average Accuracy: %.2f with a std of %.2f\n" % (best_f1_accuracy_per_cv_set.mean(), best_f1_accuracy_per_cv_set.std()))
				fp.write(">> Average Precision: %.2f with a std of %.2f\n" % (best_f1_precision_per_cv_set.mean(), best_f1_precision_per_cv_set.std()))
				fp.write(">> Average Recall: %.2f with a std of %.2f\n" % (best_f1_recall_per_cv_set.mean(), best_f1_recall_per_cv_set.std()))
				fp.write(">> Average F1: %.2f with a std of %.2f\n" % (best_f1_per_cv_set.mean(), best_f1_per_cv_set.std()))
				fp.write(">> Evaluation using the best epoch\n====================================================\n\n")
		
		elif args.eval_value == 'last':
			last_f1_accuracy_per_cv_set = np.array(last_f1_accuracy_per_cv_set)
			last_f1_precision_per_cv_set = np.array(last_f1_precision_per_cv_set)
			last_f1_recall_per_cv_set = np.array(last_f1_recall_per_cv_set)
			last_f1_per_cv_set = np.array(last_f1_per_cv_set)
			
			print(">> Average Accuracy: %.2f with a std of %.2f\n" % (last_f1_accuracy_per_cv_set.mean(), last_f1_accuracy_per_cv_set.std()))
			print(">> Average Precision: %.2f with a std of %.2f\n" % (last_f1_precision_per_cv_set.mean(), last_f1_precision_per_cv_set.std()))
			print(">> Average Recall: %.2f with a std of %.2f\n" % (last_f1_recall_per_cv_set.mean(), last_f1_recall_per_cv_set.std()))
			print(">> Average F1: %.2f with a std of %.2f\n" % (last_f1_per_cv_set.mean(), last_f1_per_cv_set.std()))
			
			with open(result_file, 'a') as fp:
				fp.write(">> Average Accuracy: %.2f with a std of %.2f\n" % (last_f1_accuracy_per_cv_set.mean(), last_f1_accuracy_per_cv_set.std()))
				fp.write(">> Average Precision: %.2f with a std of %.2f\n" % (last_f1_precision_per_cv_set.mean(), last_f1_precision_per_cv_set.std()))
				fp.write(">> Average Recall: %.2f with a std of %.2f\n" % (last_f1_recall_per_cv_set.mean(), last_f1_recall_per_cv_set.std()))
				fp.write(">> Average F1: %.2f with a std of %.2f\n" % (last_f1_per_cv_set.mean(), last_f1_per_cv_set.std()))
				fp.write(">> Evaluation using the last epoch\n====================================================\n\n")
		
		# [GP][END] - set the number of datasets for training and evaluation, and save CV results. 11-29-2020
		
	if (args.infer == 1) and (args.task != 'fewrel'):
		inferer = infer_from_trained(args, detect_entities=True)
		#test = "The surprise [E1]visit[/E1] caused a [E2]frenzy[/E2] on the already chaotic trading floor."
		test = "[E1]PCLI[/E1] was also significantly correlated to [E2]FOXM1[/E2] and CKS1B expression , but at a lower extent ."
		inferer.infer_sentence(test, detect_entities=False)
		test = "[E1]PCLI[/E1] was also significantly correlated to FOXM1 and [E2]CKS1B[/E2] expression , but at a lower extent ."
		inferer.infer_sentence(test, detect_entities=False)
		#test2 = "After eating the chicken, he developed a sore throat the next morning."
		#inferer.infer_sentence(test2, detect_entities=True)
		
		pos_pairs = {}
		neg_pairs = {}
		with open(args.test_data) as fp:
			lines = fp.readlines()
			for sent in lines:
			
				pair = [x.upper() for x in sent.split() if '[/E' in x]
				pair = [x.replace('[E1]', '').replace('[/E1]', '').replace('[E2]', '').replace('[/E2]', '') for x in pair]
				pair.sort()
				pair = '-'.join(pair)
			
				pred = inferer.infer_sentence(sent, detect_entities=False)
				
				#print(pair)
				#print(pred)
				
				if pred == 0:
					if pair in pos_pairs:
						pos_pairs[pair] += 1
					else:
						pos_pairs[pair] = 1
				elif pred == 1:
					if pair in neg_pairs:
						neg_pairs[pair] += 1
					else:
						neg_pairs[pair] = 1
		
		with open(os.path.join(args.result_dir, 'pos_ppi_count.csv'), mode='w') as f:
			fw = csv.writer(f)
			fw.writerow(['PPI', 'Count'])

			for ppi, cnt in pos_pairs.items():
				fw.writerow([ppi, cnt])
		
		with open(os.path.join(args.result_dir, 'neg_ppi_count.csv'), mode='w') as f:
			fw = csv.writer(f)
			fw.writerow(['PPI', 'Count'])

			for ppi, cnt in neg_pairs.items():
				fw.writerow([ppi, cnt])
		
		'''
		while True:
			sent = input("Type input sentence ('quit' or 'exit' to terminate):\n")
			if sent.lower() in ['quit', 'exit']:
				break
			inferer.infer_sentence(sent, detect_entities=False)
		'''
	if args.task == 'fewrel':
		fewrel = FewRel(args)
		meta_input, e1_e2_start, meta_labels, outputs = fewrel.evaluate()