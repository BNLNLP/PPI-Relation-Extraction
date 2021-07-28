#!/bin/bash

PS3='Please enter your choice: '
options=("Convert data into IOB2" 
		 "Split data" 
		 "Run NER"
		 "Run Multi Task Learning"
		 "Quit")
select opt in "${options[@]}"
do
	case $opt in
        "Convert data into IOB2")
            echo "you chose Convert data into IOB2."
			
			#export DATA_DIR=~/BER-NLP/NER/BERT-NER/dataset/BER-NER_v2-2020-10-27
			export DATA_DIR=/hpcgpfs01/scratch/gpark/archive/RadBio
			
			export PMCID_file=/hpcgpfs01/scratch/gpark/archive/RadBio/pmid-pmcid-doi.csv
			
			export OUTPUT=selected_files_iob2_format.pkl

			python ~/BER-NLP/NER/BERT-NER/data_format_converter.py \
				--data_dir=$DATA_DIR \
				--output=$OUTPUT \
				--data_type=Article
				
				: '
					--data_type=Doccano --> Doccano, Article
					--pmcid_file --> For Article, a list of PMC IDs to be considered
				'
				
			break
            ;;
        "Split data")
            echo "you chose Split data."
			
			export DATA_V1=~/BER-NLP/NER/BERT-NER/dataset/BER-NER_v1-2020-06-22/iob2_format.pkl
			export DATA_V2=~/BER-NLP/NER/BERT-NER/dataset/BER-NER_v2-2020-10-27/iob2_format.pkl
			export OUTPUT_DIR=~/BER-NLP/NER/BERT-NER/dataset/BER-NER_v1_v2
			
			python ~/BER-NLP/NER/BERT-NER/data_splitter.py \
				--data_v1=$DATA_V1 \
				--data_v2=$DATA_V2 \
				--output_dir=$OUTPUT_DIR

			break
            ;;
        "Run NER")
            echo "you chose Run NER."
			
			export MAX_LENGTH=256
			export BATCH_SIZE=32 # 32, 1 for XLNet
			export NUM_EPOCHS=3
			export SAVE_STEPS=750
			export SEED=1

			#export BERT_MODEL=bert-large-cased
			#export BERT_MODEL=bert-base-uncased
			#export BERT_MODEL=bert-base-multilingual-cased
			#export BERT_MODEL=/hpcgpfs01/scratch/gpark/bert_model/scibert/huggingface/scibert_scivocab_cased
			#export BERT_MODEL=/hpcgpfs01/scratch/gpark/bert_model/biobert/biobert_v1.1_pubmed_pytorch
			#export BERT_MODEL=dmis-lab/biobert-base-cased-v1.1
			#export BERT_MODEL=dmis-lab/biobert-large-cased-v1.1
			#export BERT_MODEL=emilyalsentzer/Bio_ClinicalBERT
			#export BERT_MODEL=roberta-large
			#export BERT_MODEL=xlnet-base-cased	# Set BATCH_SIZE=1, still not working...memory error occurs.
			#export BERT_MODEL=albert-large-v2
			#export BERT_MODEL=albert-xxlarge-v2
			#export BERT_MODEL=gpt2	# error occurs
			#export BERT_MODEL=distilbert-base-cased
			#export BERT_MODEL=dbmdz/electra-large-discriminator-finetuned-conll03-english	# not working.
			export BERT_MODEL=/hpcgpfs01/scratch/gpark/bert_model/NER/trial_5_spacy_label_error_fixed_but_forgot_to_add_remaining/dmis-lab/biobert-large-cased-v1.1
			
			#export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/bert_model/NER/trial_5_spacy_label_error_fixed_but_forgot_to_add_remaining/
			export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/bert_model/NER/trial_5_spacy_label_error_fixed_but_forgot_to_add_remaining/dmis-lab/biobert-large-cased-v1.1/RadBio/

			#export DATA_DIR=~/BER-NLP/NER/BERT-NER/dataset/BER-NER_v1_v2/trial_5_spacy_label_error_fixed_but_forgot_to_add_remaining
			export DATA_DIR=/hpcgpfs01/scratch/gpark/archive/RadBio
			
			export LABELS=~/BER-NLP/NER/BERT-NER/dataset/BER-NER_v1_v2/labels.txt

			srun -p volta -A covid-19-volta -t 24:00:00 -N 1 --gres=gpu:8 -J ner \
			python ~/BER-NLP/NER/BERT-NER/run_ner.py \
				--data_dir $DATA_DIR \
				--labels $LABELS \
				--model_name_or_path $BERT_MODEL \
				--output_dir $OUTPUT_DIR \
				--max_seq_length $MAX_LENGTH \
				--num_train_epochs $NUM_EPOCHS \
				--per_device_train_batch_size $BATCH_SIZE \
				--save_steps $SAVE_STEPS \
				--seed $SEED \
				--do_predict \
				--overwrite_cache \
				--overwrite_output_dir
				
				: '
					--do_train \
					--do_eval \
					--test_all_models
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
			
			#export NER_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/original/AImed/all_ver_1
			#export NER_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/original/BioInfer/all
			#export NER_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/original/HPRD50/all_ver_2
			#export NER_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/original/IEPA/all
			#export NER_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/original/LLL/all_ver_4
			#export NER_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/type_annotation/ALL/all
			#export NER_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/type_annotation/ALL/all_incl_only_AImed_negative
			#export NER_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/type_annotation/ALL/all_incl_only_AImed_HPRD50_IEPA_negative
			#export NER_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/type_annotation/ALL/all_incl_only_AImed_ikb_HPRD50_IEPA_negative
			#export NER_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/type_annotation/ALL/all_incl_negative_annotation_ver_6
			#export NER_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/type_annotation/ALL/all_incl_negative_annotation_ver_7
			#export NER_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/type_annotation/ALL/all_incl_negative_annotation_ver_11
			#export NER_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/type_annotation/ALL/all_incl_negative_annotation_ver_12
			#export NER_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/type_annotation/ALL/all_incl_negative_annotation_ver_15
			export NER_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/type_annotation/ALL/all_incl_negative_annotation_ver_17
			#export NER_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/type_annotation/AImed_type/all_with_negative_annotation
			
			#export PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/ppi/original/AImed/all_ver_1
			#export PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/ppi/original/BioInfer/all
			#export PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/ppi/original/HPRD50/all_ver_2
			#export PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/ppi/original/IEPA/all
			#export PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/ppi/original/LLL/all_ver_4
			#export PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/ppi/type_annotation/ALL/all
			#export PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/ppi/type_annotation/ALL/all_incl_only_AImed_negative
			#export PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/ppi/type_annotation/ALL/all_incl_only_AImed_HPRD50_IEPA_negative
			#export PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/ppi/type_annotation/ALL/all_incl_only_AImed_ikb_HPRD50_IEPA_negative
			#export PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/ppi/type_annotation/ALL/all_incl_negative_annotation_ver_6
			#export PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/ppi/type_annotation/ALL/all_incl_negative_annotation_ver_7
			#export PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/ppi/type_annotation/ALL/all_incl_negative_annotation_ver_11
			#export PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/ppi/type_annotation/ALL/all_incl_negative_annotation_ver_12
			#export PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/ppi/type_annotation/ALL/all_incl_negative_annotation_ver_15
			export PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/ppi/type_annotation/ALL/all_incl_negative_annotation_ver_17
			#export PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/ppi/type_annotation/AImed_type/all_with_negative_annotation
			
			#export JOINT_NER_PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/original/AImed/all
			#export JOINT_NER_PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/original/BioInfer/all
			#export JOINT_NER_PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/original/HPRD50/all
			#export JOINT_NER_PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/original/IEPA/all
			#export JOINT_NER_PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/original/LLL/all
			#export JOINT_NER_PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/type_annotation/ALL/all
			#export JOINT_NER_PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/type_annotation/ALL/all_incl_only_AImed_negative
			#export JOINT_NER_PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/type_annotation/ALL/all_incl_only_AImed_HPRD50_IEPA_negative
			#export JOINT_NER_PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/type_annotation/ALL/all_incl_only_AImed_ikb_HPRD50_IEPA_negative
			#export JOINT_NER_PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/type_annotation/ALL/all_incl_negative_annotation_ver_6
			#export JOINT_NER_PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/type_annotation/ALL/all_incl_negative_annotation_ver_7
			#export JOINT_NER_PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/type_annotation/ALL/all_incl_negative_annotation_ver_11
			#export JOINT_NER_PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/type_annotation/ALL/all_incl_negative_annotation_ver_12
			#export JOINT_NER_PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/type_annotation/ALL/all_incl_negative_annotation_ver_15
			export JOINT_NER_PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/type_annotation/ALL/all_incl_negative_annotation_ver_17
			#export JOINT_NER_PPI_DATA=~/BER-NLP/Multi-Task-Learning/datasets/joint_ner_ppi/type_annotation/AImed_type/all_with_negative_annotation
			
			export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/bert_model/Multi-Task-Learning/result/ppi_incl_negative_annotation_ver_17
			#export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/bert_model/Multi-Task-Learning/result/AImed/ver_1/orig_AImed_ppi_EM_entity_start
			#export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/bert_model/Multi-Task-Learning/result/BioInfer/orig_BioInfer_ner
			#export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/bert_model/Multi-Task-Learning/result/HPRD50/ver_2/orig_HPRD50_ner
			#export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/bert_model/Multi-Task-Learning/result/IEPA/orig_IEPA_ner
			#export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/bert_model/Multi-Task-Learning/result/LLL/ver_4/orig_LLL_ner
			#export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/bert_model/Multi-Task-Learning/result/original_with_bnl_ner/LLL
			#export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/bert_model/Multi-Task-Learning/result/separate_data_for_ner_ppi_tasks_no_em
			#export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/bert_model/Multi-Task-Learning/result/ppi_data_for_both_ner_ppi_tasks_no_em
			#export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/bert_model/Multi-Task-Learning/result/ppi_data_for_both_ner_ppi_tasks_with_ner_filtering


			#srun -p voltadebug -A nlp-sbu -t 4:00:00 -N 1 --gres=gpu:2 -J mtl \
			
			#srun -p volta -A covid-19-volta -t 24:00:00 -N 1 --gres=gpu:2 -J mtl \
			
			#srun -p volta -A covid-19-volta -t 24:00:00 -N 1 --gres=gpu:2 -J mtl \
			
			python ~/BER-NLP/Multi-Task-Learning/run_mt_nn.py \
				--model_list bert-base-cased dmis-lab/biobert-base-cased-v1.1 \
				--task_list ner ppi \
				--do_lower_case=False \
				--ner_data $NER_DATA \
				--ppi_data $PPI_DATA \
				--joint_ner_ppi_data $JOINT_NER_PPI_DATA \
				--ppi_classes enzyme structural negative \
				--output_dir $OUTPUT_DIR \
				--do_train \
				--do_fine_tune=True \
				--do_predict \
				--num_train_epochs=10 \
				--do_cross_validation \
				--num_of_folds=10 \
				--relation_representation EM_entity_start \
				--remove_unused_columns=False \
				--seed $SEED \
				--save_steps=9000 \
				--overwrite_cache \
				--overwrite_output_dir \
				--save_misclassified_samples

				: '
				    --model_list bert-base-cased bert-large-cased dmis-lab/biobert-base-cased-v1.1 dmis-lab/biobert-large-cased-v1.1 \
					--task_list ner ppi joint-ner-ppi \ -> do not forget to use STANDARD_mention_pooling or STANDARD_mention_pooling_plus_context for joint-ner-ppi
					--train_file $TRAIN_DATA \
					--validation_file $VALIDATION_DATA \
					--test_file $TEST_DATA \
					--ppi_classes enzyme structural negative \ --> enzyme structural negative, positive negative
					--do_train \
					--do_eval \
					--do_predict \
					--per_device_train_batch_size=16 \
					--per_device_eval_batch_size=16 \
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
					--save_misclassified_samples
				'
				
			break
            ;;
		"Quit")
            break
            ;;
        *) echo "invalid option $REPLY";;
    esac
done