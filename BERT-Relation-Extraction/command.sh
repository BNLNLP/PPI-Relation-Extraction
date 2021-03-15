#!/bin/bash

PS3='Please enter your choice: '
options=("Pretraining" 
		 "Fine-tuning" 
		 "Quit")
select opt in "${options[@]}"
do
	case $opt in
        "Pretraining")
            echo "you chose Pretraining"
			
			#export DATA_DIR=/hpcgpfs01/scratch/gpark/bert_model/BERT-Relation-Extraction/cnn/cnn.txt
			export DATA_DIR=~/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/BioCreative/PMtask_Relations_TrainingSet.json			
			
			srun -p volta -A covid-19-volta -t 24:00:00 -N 1 --gres=gpu:8 -J re \
			python ~/BER-NLP/BERT-Relation-Extraction/main_pretraining.py \
				--pretrain_data=$DATA_DIR \
				--batch_size=16 \
				--model_no=2
			
			# --batch_size=32	# out of memory error for BioCreative data
			# --model_no=0	 	# 0 - BERT 1 - ALBERT 2 - BioBERT
			
			break
            ;;
        "Fine-tuning")
            echo "you chose Fine-tuning."
			
			#export TRAIN_DIR=~/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT
			#export TEST_DIR=~/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT
			#export TRAIN_DIR=~/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/BioCreative/50_unshuffled/sean_annotation_biocreative_training_set_12_12_2020.csv
			#export TEST_DIR=~/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/BioCreative/50_unshuffled/sean_annotation_biocreative_training_set_12_12_2020.csv

			#export TRAIN_DIR=~/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/BioCreative/12_22_2020/IKB_Sean_BioCreative_full_text_annotation.txt
			#export TEST_DIR=~/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/BioCreative/12_22_2020/IKB_Sean_BioCreative_full_text_annotation.txt
			#export TRAIN_DIR=~/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/BioCreative/01_04_2021_with_biocreative_test_set/IKB_Sean_BioCreative_full_text_annotation.txt
			#export TRAIN_DIR=~/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/BioCreative/01_04_2021_with_biocreative_test_set/PMtask_Relations_TrainingSet.json
			#export TEST_DIR=~/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/BioCreative/PMtask_Relation_TestSet.json
			
			export TRAIN_DIR=~/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/AImed_BioInfer_HPRD50_IEPA_LLL/AIMed_BioInfer_HPRD50_IEPA_LLL.xml
			export TEST_DIR=~/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/AImed_BioInfer_HPRD50_IEPA_LLL/AIMed_BioInfer_HPRD50_IEPA_LLL.xml
			
			export TEST_DIR=~/RadBio/radbio_ppi.txt


			#export RESULT_DIR=~/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/results/BioCreative/12_22_2020
			export RESULT_DIR=~/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/results/PPI/AImed_BioInfer_HPRD50_IEPA_LLL
			
			export RESULT_DIR=~/RadBio
			
			export RM_FILE=PPI/AImed_BioInfer_HPRD50_IEPA_LLL/all/relations_0.pkl

			srun -p volta -A covid-19-volta -t 24:00:00 -N 1 --gres=gpu:8 -J re \
			python ~/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/main_task.py \
				--task=AImed_BioInfer_HPRD50_IEPA_LLL \
				--train_data=$TRAIN_DIR \
				--test_data=$TEST_DIR \
				--use_pretrained_blanks=0 \
				--batch_size=2 \
				--model_no=2 \
				--model_size=bert-base-uncased \
				--train=0 \
				--infer=1 \
				--rm_file=$RM_FILE \
				--num_classes=2 \
				--eval_value=last \
				--result_dir=$RESULT_DIR
			
			: ' 
			--task=BioCreative \ semeval, fewrel, BioCreative, BioCreative_BNL, AIMed, BioInfer, HPRD50, IEPA, LLL, AImed_BioInfer_HPRD50_IEPA_LLL
			--batch_size=2 \
			--num_epochs=30 \ --> default=11
			--model_size=albert-base-v2 \ --> default=bert-base-uncased, albert-base-v2
			--do_cross_validation \
			--num_of_folds=10 \
			--ratio=80-10-10 \ --> train/dev/test ratio: 80-10-10, 70-15-15, 60-20-20
			--num_samples=100 \
			--num_classes=2 \
			--do_one_class_classification \ "Whether to use sigmoid for outputs rather than softmax. this is used for testing BioCreative PPIm label.
			--threshold=0.5 \ "threshold value to determine one-class outputs."	
			--predefined_cls=PPIm \ "Whether to use predefined labels such as PPIm."			
			--eval_value=last \ --> best: get the best value among epochs, last: get the value of last epoch
			--rm_file=$RM_FILE \ --> (for infer) relation mapper path
			'
			
			break
            ;;
		"Quit")
            break
            ;;
        *) echo "invalid option $REPLY";;
    esac
done