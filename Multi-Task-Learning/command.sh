#!/bin/bash

PS3='Please enter your choice: '
options=("Pre-process datasets"
		 "Run Multi Task Learning"
		 "Quit")
select opt in "${options[@]}"
do
	case $opt in
        "Pre-process datasets")
            echo "you chose Pre-process datasets."
			
			# Original datasets (labels: positive, negative)
			#export DATA1=~/BER-NLP/Multi-Task-Learning/datasets/ppi/original/AImed/AImed.xml
			#export DATA1=~/BER-NLP/Multi-Task-Learning/datasets/ppi/original/BioInfer/BioInferCLAnalysis_split_SMBM_version.xml
			#export DATA1=~/BER-NLP/Multi-Task-Learning/datasets/ppi/original/HPRD50/HPRD50.xml			
			#export DATA1=~/BER-NLP/Multi-Task-Learning/datasets/ppi/original/IEPA/IEPA.xml
			export DATA1=~/BER-NLP/Multi-Task-Learning/datasets/ppi/original/LLL/LLL.xml
			
			# Type annotated datasets (labels: enzyme, structural, negative) - DATA1: positive (enzyme, structurel), DATA2: negative
			# Negative annotations - WARNING!! DO NOT forget to change the code in preprocessing_funcs.py (set neg_sample_processing = True) !!! 

			#export DATA1=~/BER-NLP/Multi-Task-Learning/datasets/ppi/type_annotation/BioCreative_type/01_27_2021/IKB_Sean_BioCreative_full_text_annotation.txt
			
			#export DATA1=~/BER-NLP/Multi-Task-Learning/datasets/ppi/type_annotation/AImed_type/aimed_type_annotations_srm.tsv
			#export DATA2=~/BER-NLP/Multi-Task-Learning/datasets/ppi/type_annotation/AImed_type/passed_full_aimed_training_ikb.tsv
			
			#export DATA1=~/BER-NLP/Multi-Task-Learning/datasets/ppi/type_annotation/BioInfer_type/bioinfer_type_annotations_struct_histones_srm.tsv
			#export DATA2=~/BER-NLP/Multi-Task-Learning/datasets/ppi/type_annotation/BioInfer_type/passed_full_bioinfer_training.tsv
			
			#export DATA1=~/BER-NLP/Multi-Task-Learning/datasets/ppi/type_annotation/HPRD50_type/hprd50_type_annotation_srm.tsv
			#export DATA2=~/BER-NLP/Multi-Task-Learning/datasets/ppi/type_annotation/HPRD50_type/passed_full_hprd50_training.tsv
			
			#export DATA1=~/BER-NLP/Multi-Task-Learning/datasets/ppi/type_annotation/IEPA_type/iepa_type_annotations_srm.tsv
			#export DATA2=~/BER-NLP/Multi-Task-Learning/datasets/ppi/type_annotation/IEPA_type/passed_full_iepa_training.tsv
			
			#export DATA1=~/BER-NLP/Multi-Task-Learning/datasets/ppi/type_annotation/LLL_type/lll_type_annotations_srm.tsv
			
			# Positive (enzyme, structural) annotations - filtered error cases (duplicate labeling for the same pair in a sentence e.g., A-B: enzyme, A-B: negative) 08-31-2021
			#export DATA1=~/BER-NLP/Multi-Task-Learning/datasets/ppi/type_annotation/AImed_type/aimed_type_annotations_srm.filtered.tsv
			#export DATA1=~/BER-NLP/Multi-Task-Learning/datasets/ppi/type_annotation/BioInfer_type/bioinfer_type_annotations_struct_histones_srm.filtered.tsv
			#export DATA1=~/BER-NLP/Multi-Task-Learning/datasets/ppi/type_annotation/HPRD50_type/hprd50_type_annotation_srm.filtered.tsv
			#export DATA1=~/BER-NLP/Multi-Task-Learning/datasets/ppi/type_annotation/IEPA_type/iepa_type_annotations_srm.filtered.tsv
			#export DATA1=~/BER-NLP/Multi-Task-Learning/datasets/ppi/type_annotation/LLL_type/lll_type_annotations_srm.filtered.tsv
			

			python ~/BER-NLP/Multi-Task-Learning/preprocessor.py \
				--data_list $DATA1 \
				--task=PPIbenchmark \
				--do_cross_validation \
				--num_of_folds=10 \
				--ratio=k-folds

			: ' 
			--task=PPIbenchmark \ --> PPIbenchmark, PPIbenchmark_type, BioCreative_type, BioCreative (not used for now)
			--do_cross_validation \
			--num_of_folds=10 \
			--ratio=k-folds, 80-10-10 \ --> k-folds: train/test given the number of folds, train/dev/test ratio: 80-10-10, 70-15-15, 60-20-20
			--num_samples=100 \
			'
				
			break
            ;;
        "Run Multi Task Learning")
            echo "you chose Run Multi Task Learning."
			
			#export MAX_LENGTH=256
			#export BATCH_SIZE=32 # 32, 1 for XLNet
			#export NUM_EPOCHS=3
			#export SAVE_STEPS=750
			export SEED=1

			#export MODEL=bert-large-cased
			#export MODEL=bert-base-cased
			#export MODEL=bert-base-uncased
			#export MODEL=bert-base-multilingual-cased
			#export MODEL=/hpcgpfs01/scratch/gpark/bert_model/scibert/huggingface/scibert_scivocab_cased
			#export MODEL=/hpcgpfs01/scratch/gpark/bert_model/biobert/biobert_v1.1_pubmed_pytorch
			#export MODEL=dmis-lab/biobert-base-cased-v1.1
			#export MODEL=dmis-lab/biobert-large-cased-v1.1
			#export MODEL=emilyalsentzer/Bio_ClinicalBERT
			#export MODEL=roberta-large
			#export MODEL=xlnet-base-cased	# Set BATCH_SIZE=1, still not working...memory error occurs.
			#export MODEL=albert-large-v2
			#export MODEL=albert-xxlarge-v2
			#export MODEL=gpt2	# error occurs
			#export MODEL=distilbert-base-cased
			#export MODEL=dbmdz/electra-large-discriminator-finetuned-conll03-english	# not working.
			#export MODEL=/hpcgpfs01/scratch/gpark/bert_model/NER/trial_5_spacy_label_error_fixed_but_forgot_to_add_remaining/dmis-lab/biobert-large-cased-v1.1

			#export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/bert_model/NER/trial_5_spacy_label_error_fixed_but_forgot_to_add_remaining/
			#export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/bert_model/NER_new/

			#export DATA_DIR=~/BER-NLP/NER/BERT-NER/dataset/BER-NER_v1_v2/trial_5_spacy_label_error_fixed_but_forgot_to_add_remaining
			#export DATA_DIR=/hpcgpfs01/scratch/gpark/archive/RadBio
			
			#export LABELS=~/BER-NLP/NER/BERT-NER/dataset/BER-NER_v1_v2/labels.txt

			#export TRAIN_DATA=~/BER-NLP/PPI-Relation-Extraction/Multi-Task-Learning/new_ner_code_not_working/data/train.json
			#export VALIDATION_DATA=~/BER-NLP/PPI-Relation-Extraction/Multi-Task-Learning/new_ner_code_not_working/data/dev.json
			#export TEST_DATA=~/BER-NLP/PPI-Relation-Extraction/Multi-Task-Learning/new_ner_code_not_working/data/test.json
			
			#export NER_DATA=~/BER-NLP/Multi-Task-Learning/datasets/ner
			#export NER_DATA=~/BER-NLP/Multi-Task-Learning/datasets/ner/10_fold_cv
			#export NER_DATA=~/BER-NLP/Multi-Task-Learning/datasets/ner/BER-NLP_v1_v2_trial_5_spacy_label_error_fixed_but_forgot_to_add_remaining_10_fold_cv
			#export NER_DATA=~/BER-NLP/Multi-Task-Learning/datasets/ner/BER-NLP_v1_v2_trial_5_spacy_label_error_fixed_but_forgot_to_add_remaining_10_fold_cv_only_protein
			#export NER_DATA=~/BER-NLP/Multi-Task-Learning/datasets/ner/BioCreative_I_Gene_Mention_Identification_10_fold_cv_shuffled
			#export NER_DATA=~/BER-NLP/Multi-Task-Learning/datasets/ner/BioCreative_I_Gene_Mention_Identification_10_fold_cv_no_test_data
			#export NER_DATA=~/BER-NLP/Multi-Task-Learning/datasets/ner/BER_NLP_v1_v2_trial_5_and_BioCreative_I_Gene_Mention_Identification_10_fold_cv
			#export NER_DATA=~/BER-NLP/Multi-Task-Learning/datasets/ner/BER_NLP_v1_v2_trial_5_only_protein_and_BioCreative_I_Gene_Mention_Identification_no_test_data_10_fold_cv

			#export NER_DATA=~/BER-NLP/Multi-Task-Learning/datasets/ppi/type_annotation/ALL/all
			#export NER_DATA=~/BER-NLP/Multi-Task-Learning/datasets/ppi/original/LLL/all
			
			#export NER_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/original/AImed/all_ver_3
			#export NER_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/original/BioInfer/all
			#export NER_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/original/HPRD50/all_ver_2
			#export NER_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/original/IEPA/all_ver_1
			#export NER_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/original/LLL/all_ver_5
			#export NER_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/original/LLL/all_ver_5_separate
			#export NER_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/type_annotation/ALL/all
			#export NER_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/type_annotation/ALL/all_incl_only_AImed_negative
			#export NER_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/type_annotation/ALL/all_incl_only_AImed_HPRD50_IEPA_negative
			#export NER_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/type_annotation/ALL/all_incl_only_AImed_ikb_HPRD50_IEPA_negative
			export NER_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/type_annotation/ALL/all_incl_negative_annotation_ver_17
			#export NER_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/type_annotation/AImed_type/all_with_negative_annotation
			
			#export PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/ppi/original/AImed/all_ver_3
			#export PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/ppi/original/BioInfer/all
			#export PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/ppi/original/HPRD50/all_ver_2
			#export PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/ppi/original/IEPA/all_ver_1
			#export PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/ppi/original/LLL/all_ver_5
			#export PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/ppi/type_annotation/ALL/all
			#export PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/ppi/type_annotation/ALL/all_incl_only_AImed_negative
			#export PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/ppi/type_annotation/ALL/all_incl_only_AImed_HPRD50_IEPA_negative
			#export PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/ppi/type_annotation/ALL/all_incl_only_AImed_ikb_HPRD50_IEPA_negative
			#export PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/ppi/type_annotation/ALL/all_incl_negative_annotation_ver_17
			#export PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/ppi/type_annotation/AImed_type/all_with_negative_annotation
			#export PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/ade/converted
			export PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/conll04/converted
			
			#export JOINT_NER_PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/original/AImed/all_ver_3
			#export JOINT_NER_PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/original/BioInfer/all
			#export JOINT_NER_PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/original/HPRD50/all_ver_2
			#export JOINT_NER_PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/original/IEPA/all_ver_1
			#export JOINT_NER_PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/original/LLL/all_ver_5
			export JOINT_NER_PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/original/LLL/all_ver_5_separate
			#export JOINT_NER_PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/type_annotation/ALL/all
			#export JOINT_NER_PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/type_annotation/ALL/all_incl_only_AImed_negative
			#export JOINT_NER_PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/type_annotation/ALL/all_incl_only_AImed_HPRD50_IEPA_negative
			#export JOINT_NER_PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/type_annotation/ALL/all_incl_only_AImed_ikb_HPRD50_IEPA_negative
			#export JOINT_NER_PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/type_annotation/ALL/all_incl_negative_annotation_ver_17
			#export JOINT_NER_PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/type_annotation/AImed_type/all_with_negative_annotation
			
			#export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/bert_model/Multi-Task-Learning/result/ppi_incl_negative_annotation_ver_17_transformers_ver_4_12_0_json_adding_both_directions_2
			#export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/bert_model/Multi-Task-Learning/result/ppi_incl_negative_annotation_ver_17_transformers_ver_4_12_0
			#export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/bert_model/Multi-Task-Learning/result/AImed/ver_3/orig_AImed_mtl_EM_entity_start_transformers_ver_4_12_0
			#export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/bert_model/Multi-Task-Learning/result/BioInfer/ver_3/orig_BioInfer_ner
			#export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/bert_model/Multi-Task-Learning/result/HPRD50/ver_2/orig_HPRD50_ner
			#export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/bert_model/Multi-Task-Learning/result/IEPA/ver_1/orig_IEPA_ner
			#export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/bert_model/Multi-Task-Learning/result/LLL/ver_5/orig_LLL_ppi_EM_entity_start_transformers_ver_4_12_0_json_adding_both_directions
			#export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/bert_model/Multi-Task-Learning/result/LLL/ver_5_separate/orig_LLL_joint_STANDARD_mention_pooling_plus_context_epoch_30
			#export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/bert_model/Multi-Task-Learning/result/original_with_bnl_ner/LLL
			#export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/bert_model/Multi-Task-Learning/result/separate_data_for_ner_ppi_tasks_no_em
			#export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/bert_model/Multi-Task-Learning/result/ppi_data_for_both_ner_ppi_tasks_no_em
			#export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/bert_model/Multi-Task-Learning/result/ppi_data_for_both_ner_ppi_tasks_with_ner_filtering
			#export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/bert_model/Multi-Task-Learning/result/ADE/ppi_EM_entity_start_transformers_ver_4_12_0_json_adding_both_directions
			export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/bert_model/Multi-Task-Learning/result/CoNLL04/ppi_STANDARD_mention_pooling_plus_context_ppp_transformers_ver_4_12_0_json_adding_both_directions
			
			#export RELATIONS=~/BER-NLP/Multi-Task-Learning/datasets/ppi/original/relations.json
			#export RELATIONS=~/BER-NLP/Multi-Task-Learning/datasets/ppi/type_annotation/relations.json
			#export RELATIONS=~/BER-NLP/Multi-Task-Learning/datasets/ade/converted/relations.json
			export RELATIONS=~/BER-NLP/Multi-Task-Learning/datasets/conll04/converted/relations.json
			
			
			#srun -p volta -A nlp-sbu -t 24:00:00 -N 1 --gres=gpu:2 -J mtl \
			#srun -p voltadebug -A nlp-sbu -t 4:00:00 -N 1 --gres=gpu:2 -J mtl \
			
			srun -p volta -A nlp-sbu -t 24:00:00 -N 1 --gres=gpu:2 -J mtl \
			python ~/BER-NLP/Multi-Task-Learning/run_mt_nn.py \
				--model_list bert-base-cased \
				--task_list ppi \
				--do_lower_case=False \
				--ner_data $NER_DATA \
				--ppi_data $PPI_DATA \
				--joint_ner_ppi_data $JOINT_NER_PPI_DATA \
				--relations $RELATIONS \
				--output_dir $OUTPUT_DIR \
				--do_train \
				--do_predict \
				--num_train_epochs=10 \
				--num_of_folds=10 \
				--relation_representation STANDARD_mention_pooling_plus_context \
				--remove_unused_columns=False \
				--seed $SEED \
				--save_steps=9000 \
				--overwrite_cache \
				--overwrite_output_dir
				
				: '
				    --model_list dmis-lab/biobert-base-cased-v1.1 bert-base-cased roberta-base allenai/biomed_roberta_base \
								 albert-base-v2 \ --> uncased model !!!
								 microsoft/deberta-base \
								 microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \ --> uncased model !!!
								 microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \ --> uncased model !!!
								 bert-large-cased dmis-lab/biobert-large-cased-v1.1 \
								 
					--task_list ner ppi joint-ner-ppi \ -> do not forget to use STANDARD_mention_pooling or STANDARD_mention_pooling_plus_context for joint-ner-ppi
					--train_file $TRAIN_DATA \
					--validation_file $VALIDATION_DATA \
					--test_file $TEST_DATA \
					--do_train \
					--do_eval \
					--do_predict \
					--per_device_train_batch_size=16 \ --> default: per_device_train_batch_size=8
					--per_device_eval_batch_size=16 \ --> default: per_device_eval_batch_size=8
					--learning_rate=0.0001 \
					--test_all_models \
					--do_lower_case=False \
					--do_cross_validation \
					--num_of_folds=10 \
					--relation_representation EM_entity_start \ ->  STANDARD_cls_token, STANDARD_mention_pooling, STANDARD_mention_pooling_plus_context, EM_cls_token, EM_mention_pooling, EM_entity_start, EM_entity_start_plus_context, Multiple_Relations \
					--load_best_model_at_end=True \
					--learning_rate=0.00007 \
					--evaluation_strategy=epoch \
					--seed $SEED \
					--save_misclassified_samples \ -> turn off for joint-ner-ppi since it causes an error. TODO: fix the error!!
				'
				
			break
            ;;
		"Quit")
            break
            ;;
        *) echo "invalid option $REPLY";;
    esac
done