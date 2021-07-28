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
			
			#export TRAIN_DATA=~/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT
			#export TEST_DATA=~/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT
			#export TRAIN_DATA=~/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/BioCreative/50_unshuffled/sean_annotation_biocreative_training_set_12_12_2020.csv
			#export TEST_DATA=~/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/BioCreative/50_unshuffled/sean_annotation_biocreative_training_set_12_12_2020.csv

			#export TRAIN_DATA=~/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/BioCreative/12_22_2020/IKB_Sean_BioCreative_full_text_annotation.txt
			#export TEST_DATA=~/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/BioCreative/12_22_2020/IKB_Sean_BioCreative_full_text_annotation.txt
			#export TRAIN_DATA=~/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/BioCreative/01_04_2021_with_biocreative_test_set/IKB_Sean_BioCreative_full_text_annotation.txt
			#export TRAIN_DATA=~/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/BioCreative/01_04_2021_with_biocreative_test_set/PMtask_Relations_TrainingSet.json
			#export TRAIN_DATA=~/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/BioCreative/01_04_2021_with_biocreative_test_set/PMtask_Relations_TrainingSet.json
			#export TEST_DATA=~/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/BioCreative/PMtask_Relation_TestSet.json
			
			#export TRAIN_DATA=~/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/BioInfer_type/bioinfer_type_annotations_srm.tsv
			#export TEST_DATA=~/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/BioInfer_type/bioinfer_type_annotations_srm.tsv
			
			# Original datasets
			#export TRAIN_DATA=~/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/original/AImed/AImed.xml
			#export TRAIN_DATA=~/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/original/BioInfer/BioInferCLAnalysis_split_SMBM_version.xml
			#export TRAIN_DATA=~/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/original/HPRD50/HPRD50.xml			
			#export TRAIN_DATA=~/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/original/IEPA/IEPA.xml
			#export TRAIN_DATA=~/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/original/LLL/LLL.xml
			
			# Positive (enzyme, structural) annotations
			export TRAIN_DATA=~/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/type_annotation/BioCreative_type/01_27_2021/IKB_Sean_BioCreative_full_text_annotation.txt
			#export TRAIN_DATA=~/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/type_annotation/AImed_type/aimed_type_annotations_srm.tsv
			#export TRAIN_DATA=~/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/type_annotation/BioInfer_type/bioinfer_type_annotations_struct_histones_srm.tsv
			#export TRAIN_DATA=~/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/type_annotation/HPRD50_type/hprd50_type_annotation_srm.tsv
			#export TRAIN_DATA=~/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/type_annotation/IEPA_type/iepa_type_annotations_srm.tsv
			#export TRAIN_DATA=~/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/type_annotation/LLL_type/lll_type_annotations_srm.tsv
			
			# Negative annotations - WARNING!! DO NOT forget to change the code in preprocessing_funcs.py (set neg_sample_processing = True) !!! 
			#export TRAIN_DATA=~/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/type_annotation/AImed_type/passed_full_aimed_training_ikb.tsv
			#export TRAIN_DATA=~/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/type_annotation/BioInfer_type/passed_full_bioinfer_training.tsv
			#export TRAIN_DATA=~/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/type_annotation/HPRD50_type/passed_full_hprd50_training.tsv
			#export TRAIN_DATA=~/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/type_annotation/IEPA_type/passed_full_iepa_training.tsv
		
			#export TRAIN_DATA=~/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/type_annotation/ALL/all
			export TEST_DATA=~/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/data/PPI/type_annotation/ALL/all

			#export TEST_DATA=~/RadBio/radbio_ppi.txt

			#export RESULT_DIR=~/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/results/BioCreative/12_22_2020
			#export RESULT_DIR=~/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/results/PPI/AImed_BioInfer_HPRD50_IEPA_LLL
			export RESULT_DIR=~/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/results/PPI/type_annotation
			
			#export RESULT_DIR=~/RadBio
			
			export RM_FILE=PPI/AImed_BioInfer_HPRD50_IEPA_LLL/all/relations_0.pkl

			#srun -p volta -A covid-19-volta -t 24:00:00 -N 1 --gres=gpu:8 -J re \
			
			python ~/BER-NLP/PPI-Relation-Extraction/BERT-Relation-Extraction/main_task.py \
				--task=PPIbenchmark_type \
				--num_classes=3 \
				--classes enzyme structural negative \
				--train_data=$TRAIN_DATA \
				--test_data=$TEST_DATA \
				--use_pretrained_blanks=0 \
				--batch_size=2 \
				--model_no=2 \
				--model_size=bert-base-uncased \
				--do_cross_validation \
				--num_of_folds=10 \
				--ratio=k-folds \
				--train=1 \
				--infer=0 \
				--rm_file=$RM_FILE \
				--eval_value=last \
				--result_dir=$RESULT_DIR
			
			: ' 
			--task=BioCreative \ BioCreative, BioCreative_type, 
								 PPIbenchmark, PPIbenchmark_type
			--batch_size=2 \
			--num_epochs=30 \ --> default=11
			--model_size=albert-base-v2 \ --> default=bert-base-uncased, albert-base-v2
			--do_cross_validation \
			--num_of_folds=10 \
			--ratio=k-folds, 80-10-10 \ --> k-folds: train/test given the number of folds, train/dev/test ratio: 80-10-10, 70-15-15, 60-20-20
			--num_samples=100 \
			--num_classes=3 \
			--classes enzyme structural negative \ --> enzyme structural negative, positive negative
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