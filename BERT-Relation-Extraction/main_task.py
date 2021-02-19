#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 17:40:16 2019

@author: weetee
"""
from src.tasks.trainer import train_and_fit
from src.tasks.infer import infer_from_trained, FewRel
import logging
from argparse import ArgumentParser
import os

## Find the modifications by the tag [GP].

'''
This fine-tunes the BERT model on SemEval, FewRel tasks
'''

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("--task", type=str, default='semeval', help='semeval, biocreative, fewrel') # biocreative added. 11-26-2020
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
	parser.add_argument("--do_cross_validation", action="store_true", help="Whether to use cross-validation for evaluation.")
	parser.add_argument("--num_of_folds", default=10, type=int, help="The number of folds for the cross validation.")
	parser.add_argument("--eval_value", type=str, help="best: get the best value among epochs, last: get the value of last epoch")
	parser.add_argument("--result_dir", type=str, help="result directory path")
	# [GP][END] - arguments for cross-validation. 11-29-2020
	
	args = parser.parse_args()
	
	if (args.train == 1) and (args.task != 'fewrel'):
		# [GP][START] - set the number of datasets for training and evaluation, and save CV results. 11-29-2020
		num_of_datasets = 1 # if not CV, there is only one dataset.
		if args.do_cross_validation:
			num_of_datasets = args.num_of_folds
		
		if not os.path.exists(args.result_dir):
			os.makedirs(args.result_dir)
		
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
			
		result_file = os.path.join(args.result_dir, result_file)
		
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
				fp.write("cv set: %d / accuracy: %.2f / precision: %.2f / recall: %.2f / f1: %.2f\n" % (dataset_num, result['accuracy'], result['precision'], result['recall'], result['f1']))
				print("cv set: %d / accuracy: %.2f / precision: %.2f / recall: %.2f / f1: %.2f\n" % (dataset_num, result['accuracy'], result['precision'], result['recall'], result['f1']))
						
				fp.write('--------------------------\n')
			
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
			
			# remove files for the next CV set.
			os.remove(os.path.join("./data/" , "task_test_model_best_%d.pth.tar" % args.model_no))
			os.remove(os.path.join("./data/" , "task_test_checkpoint_%d.pth.tar" % args.model_no))
			os.remove(os.path.join("./data/" , "task_test_losses_per_epoch_%d.pkl" % args.model_no))
			os.remove(os.path.join("./data/" , "task_train_accuracy_per_epoch_%d.pkl" % args.model_no))
			os.remove(os.path.join("./data/" , "task_test_f1_per_epoch_%d.pkl" % args.model_no))
			
		if args.eval_value == 'best':
			print(">> Average Accuracy: %.2f\n" % (sum(best_f1_accuracy_per_cv_set)/len(best_f1_accuracy_per_cv_set)))
			print(">> Average Precision: %.2f\n" % (sum(best_f1_precision_per_cv_set)/len(best_f1_precision_per_cv_set)))
			print(">> Average Recall: %.2f\n" % (sum(best_f1_recall_per_cv_set)/len(best_f1_recall_per_cv_set)))
			print(">> Average F1: %.2f\n" % (sum(best_f1_per_cv_set)/len(best_f1_per_cv_set)))
			
			with open(result_file, 'a') as fp:
				fp.write(">> Average Accuracy: %.2f\n" % (sum(best_f1_accuracy_per_cv_set)/len(best_f1_accuracy_per_cv_set)))
				fp.write(">> Average Precision: %.2f\n" % (sum(best_f1_precision_per_cv_set)/len(best_f1_precision_per_cv_set)))
				fp.write(">> Average Recall: %.2f\n" % (sum(best_f1_recall_per_cv_set)/len(best_f1_recall_per_cv_set)))
				fp.write(">> Average F1: %.2f\n" % (sum(best_f1_per_cv_set)/len(best_f1_per_cv_set)))
				fp.write(">> Evaluation using the best epoch\n====================================================\n\n")
		
		elif args.eval_value == 'last':
			print(">> Average Accuracy: %.2f\n" % (sum(last_f1_accuracy_per_cv_set)/len(last_f1_accuracy_per_cv_set)))
			print(">> Average Precision: %.2f\n" % (sum(last_f1_precision_per_cv_set)/len(last_f1_precision_per_cv_set)))
			print(">> Average Recall: %.2f\n" % (sum(last_f1_recall_per_cv_set)/len(last_f1_recall_per_cv_set)))
			print(">> Average F1: %.2f\n" % (sum(last_f1_per_cv_set)/len(last_f1_per_cv_set)))
			
			with open(result_file, 'a') as fp:
				fp.write(">> Average Accuracy: %.2f\n" % (sum(last_f1_accuracy_per_cv_set)/len(last_f1_accuracy_per_cv_set)))
				fp.write(">> Average Precision: %.2f\n" % (sum(last_f1_precision_per_cv_set)/len(last_f1_precision_per_cv_set)))
				fp.write(">> Average Recall: %.2f\n" % (sum(last_f1_recall_per_cv_set)/len(last_f1_recall_per_cv_set)))
				fp.write(">> Average F1: %.2f\n" % (sum(last_f1_per_cv_set)/len(last_f1_per_cv_set)))
				fp.write(">> Evaluation using the last epoch\n====================================================\n\n")
		
		# [GP][END] - set the number of datasets for training and evaluation, and save CV results. 11-29-2020
		
	if (args.infer == 1) and (args.task != 'fewrel'):
		inferer = infer_from_trained(args, detect_entities=True)
		test = "The surprise [E1]visit[/E1] caused a [E2]frenzy[/E2] on the already chaotic trading floor."
		inferer.infer_sentence(test, detect_entities=False)
		test2 = "After eating the chicken, he developed a sore throat the next morning."
		inferer.infer_sentence(test2, detect_entities=True)
		
		while True:
			sent = input("Type input sentence ('quit' or 'exit' to terminate):\n")
			if sent.lower() in ['quit', 'exit']:
				break
			inferer.infer_sentence(sent, detect_entities=False)
	
	if args.task == 'fewrel':
		fewrel = FewRel(args)
		meta_input, e1_e2_start, meta_labels, outputs = fewrel.evaluate()