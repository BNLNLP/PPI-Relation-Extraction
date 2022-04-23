#!/bin/bash

PS3='Please enter your choice: '
options=("Pre-process datasets"
		 "Run RE"
		 "Tune Model (search hyperparameter)"
		 "Converting Tensorflow Checkpoints"
		 "Quit")
select opt in "${options[@]}"
do
	case $opt in
        "Pre-process datasets")
            echo "you chose Pre-process datasets."
			
			# RE benchmark
			#export DATASET_NAME=ChemProt_BLURB
			#export DATASET_NAME=DDI_BLURB
			#export DATASET_NAME=GAD_BLURB
			#export DATASET_NAME=EU-ADR_BioBERT
			
			# PPI benchmark
			#export DATASET_NAME=AImed
			export DATASET_NAME=BioInfer
			#export DATASET_NAME=HPRD50
			#export DATASET_NAME=IEPA
			#export DATASET_NAME=LLL
			
			# Typed PPI
			#export DATASET_NAME=AImed_typed
			#export DATASET_NAME=BioInfer_typed
			#export DATASET_NAME=HPRD50_typed
			#export DATASET_NAME=IEPA_typed
			#export DATASET_NAME=LLL_typed
			
			# GRR (Gene regulatory relation)
			#export DATASET_NAME=P-putida
			
			python ~/BER-NLP/RE/datasets/conversion/data_preprocessor.py \
				--dataset_name $DATASET_NAME
				
			break
            ;;
        "Run RE")
            echo "you chose RE."
			
			#export MAX_LENGTH=256
			#export BATCH_SIZE=32 # 32, 1 for XLNet
			#export NUM_EPOCHS=3
			#export SAVE_STEPS=750
			export SEED=1

			#export DATA=~/BER-NLP/RE/datasets/PPI/original/AImed/all_best
			#export DATA=~/BER-NLP/RE/datasets/PPI/original/BioInfer/all_best
			#export DATA=~/BER-NLP/RE/datasets/PPI/original/HPRD50/all_best
			#export DATA=~/BER-NLP/RE/datasets/PPI/original/IEPA/all_best
			#export DATA=~/BER-NLP/RE/datasets/PPI/original/LLL/all_best
			#export DATA=~/BER-NLP/RE/datasets/PPI/type_annotation/ALL/all
			#export DATA=~/BER-NLP/RE/datasets/PPI/type_annotation/ALL/all_incl_only_AImed_negative
			#export DATA=~/BER-NLP/RE/datasets/PPI/type_annotation/ALL/all_incl_only_AImed_HPRD50_IEPA_negative
			#export DATA=~/BER-NLP/RE/datasets/PPI/type_annotation/ALL/all_incl_only_AImed_ikb_HPRD50_IEPA_negative
			#export DATA=~/BER-NLP/RE/datasets/PPI/type_annotation/ALL/all_incl_negative_annotation_ver_12/em_free
			#export DATA=~/BER-NLP/RE/datasets/PPI/type_annotation/AImed_type/all_with_negative_annotation
			#export DATA=~/BER-NLP/RE/datasets/ade/converted
			#export DATA=~/BER-NLP/RE/datasets/conll04/converted
			#export DATA=~/BER-NLP/RE/datasets/scierc/converted
			#export DATA=~/BER-NLP/RE/datasets/chemprot/converted
			#export DATA=~/BER-NLP/RE/datasets/chemprot/BLURB
			#export DATA=~/BER-NLP/RE/datasets/ddi/BLURB
			#export DATA=~/BER-NLP/RE/datasets/GAD
			#export DATA=~/BER-NLP/RE/datasets/gad_10-fold_CV/converted
			#export DATA=~/BER-NLP/RE/datasets/euadr/converted
			#export DATA=~/BER-NLP/RE/datasets/tacred/tacred/data/converted
			#export DATA=~/RadBio/datasets/merged_samples
			
			#export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/RE/result/Typed_PPI/ppi_incl_negative_annotation_ver_12_POSITIONAL_using_entity_type_plus_pos_emb
			#export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/RE/result/AImed/ver_best/orig_AImed_ppi_POSITIONAL_using_entity_type_plus_pos_emb
			#export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/RE/result/BioInfer/ver_best/orig_BioInfer_ppi_POSITIONAL_using_pos_emb
			#export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/RE/result/HPRD50/ver_best/orig_HPRD50_ppi_POSITIONAL_using_entity_type_plus_pos_emb
			#export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/RE/result/IEPA/ver_best/orig_IEPA_ppi_POSITIONAL_using_entity_type
			#export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/RE/result/LLL/ver_best/orig_LLL_ppi_POSITIONAL_using_pos_emb
			#export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/RE/result/LLL/ver_5_separate/orig_LLL_joint_STANDARD_mention_pooling_plus_context_epoch_30
			#export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/RE/result/original_with_bnl_ner/LLL
			#export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/RE/result/separate_data_for_ner_ppi_tasks_no_em
			#export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/RE/result/ppi_data_for_both_ner_ppi_tasks_no_em
			#export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/RE/result/ppi_data_for_both_ner_ppi_tasks_with_ner_filtering
			#export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/RE/result/ADE/ppi_EM_entity_start_transformers_ver_4_12_0_json_adding_both_directions
			#export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/RE/result/CoNLL04/ppi_POSITIONAL_mention_pooling_plus_context
			#export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/RE/result/SciERT/test
			#export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/RE/result/CHEMPROT/test
			#export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/RE/result/DDI/test
			#export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/RE/result/GAD/test
			#export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/RE/result/GAD_CV/ppi_POSITIONAL
			#export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/RE/result/EUADR/ppi_POSITIONAL_using_entity_type_plus_pos_emb
			#export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/RE/result/TACRED/ppi_POSITIONAL_gpu_8_batch_2_lr_2e_5
			#export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/RE/result/gene/ppi_using_entity_type_plus_pos_emb_single_e_type
			#export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/RE/result/pputida/ppi_using_entity_type_plus_pos_emb_single_e_type

			#export RELATION_TYPES=~/BER-NLP/RE/datasets/PPI/original/relations.json
			#export RELATION_TYPES=~/BER-NLP/RE/datasets/PPI/type_annotation/relations.json
			#export RELATION_TYPES=~/BER-NLP/RE/datasets/ade/converted/relations.json
			#export RELATION_TYPES=~/BER-NLP/RE/datasets/conll04/converted/relations.json
			#export RELATION_TYPES=~/BER-NLP/RE/datasets/scierc/converted/relation_types.json
			#export RELATION_TYPES=~/BER-NLP/RE/datasets/chemprot/converted/relation_types.json
			#export RELATION_TYPES=~/BER-NLP/RE/datasets/chemprot/BLURB/relation_types.json
			#export RELATION_TYPES=~/BER-NLP/RE/datasets/ddi/converted/relation_types.json
			#export RELATION_TYPES=~/BER-NLP/RE/datasets/ddi/BLURB/relation_types.json
			#export RELATION_TYPES=~/BER-NLP/RE/datasets/gad/relation_types.json
			#export RELATION_TYPES=~/BER-NLP/RE/datasets/euadr/converted/relation_types.json
			#export RELATION_TYPES=~/BER-NLP/RE/datasets/tacred/tacred/data/converted/relation_types.json
			#export RELATION_TYPES=~/RadBio/datasets/merged_samples/relation_types.json

			#export ENTITY_TYPES=~/BER-NLP/RE/datasets/PPI/original/AImed/entity_types.json
			#export ENTITY_TYPES=~/BER-NLP/RE/datasets/PPI/original/BioInfer/entity_types.json
			#export ENTITY_TYPES=~/BER-NLP/RE/datasets/PPI/original/HPRD50/entity_types.json
			#export ENTITY_TYPES=~/BER-NLP/RE/datasets/PPI/original/IEPA/entity_types.json
			#export ENTITY_TYPES=~/BER-NLP/RE/datasets/PPI/original/LLL/entity_types.json
			#export ENTITY_TYPES=~/BER-NLP/RE/datasets/PPI/type_annotation/entity_types.json
			#export ENTITY_TYPES=~/BER-NLP/RE/datasets/scierc/converted/entity_types.json
			#export ENTITY_TYPES=~/BER-NLP/RE/datasets/chemprot/converted/entity_types.json
			#export ENTITY_TYPES=~/BER-NLP/RE/datasets/chemprot/BLURB/entity_types.json
			#export ENTITY_TYPES=~/BER-NLP/RE/datasets/ddi/converted/entity_types.json
			#export ENTITY_TYPES=~/BER-NLP/RE/datasets/ddi/BLURB/entity_types.json
			#export ENTITY_TYPES=~/BER-NLP/RE/datasets/gad/entity_types.json
			#export ENTITY_TYPES=~/BER-NLP/RE/datasets/euadr/converted/entity_types.json
			#export ENTITY_TYPES=~/BER-NLP/RE/datasets/tacred/tacred/data/converted/entity_types.json
			#export ENTITY_TYPES=~/RadBio/datasets/merged_samples/entity_types.json

			#export TRAIN_DATA=~/RadBio/datasets/merged_samples/train_0.json
			#export TEST_DATA=~/RadBio/datasets/pputida_samples/pputida_samples.json
			#export TEST_DATA=~/RadBio/datasets/merged_samples/test_0.json
			
			
			export DATASET_DIR=~/BER-NLP/RE/datasets
			export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/RE_results
			
			# RE benchmark
			#export DATASET_NAME=ChemProt_BLURB
			#export DATASET_NAME=DDI_BLURB
			#export DATASET_NAME=GAD_BLURB
			#export DATASET_NAME=EU-ADR_BioBERT
			
			# PPI benchmark
			#export DATASET_NAME=PPI/original/AImed
			export DATASET_NAME=PPI/original/BioInfer
			#export DATASET_NAME=PPI/original/HPRD50
			#export DATASET_NAME=PPI/original/IEPA
			#export DATASET_NAME=PPI/original/LLL
			
			# Typed PPI
			#export DATASET_NAME=PPI/type_annotation/asdf
			
			# GRR (Gene regulatory relation)
			#export DATASET_NAME=GRR/P-putida

			#srun -p volta -A nlp-sbu -t 24:00:00 -N 1 --gres=gpu:2 -J re \
			#srun -p voltadebug -A nlp-sbu -t 4:00:00 -N 1 --gres=gpu:2 -J re \

			srun -p volta -A nlp-sbu -t 24:00:00 -N 1 --gres=gpu:2 -J re \
			python ~/BER-NLP/RE/re/run_re.py \
				--model_list dmis-lab/biobert-large-cased-v1.1 \
				--task_name "re" \
				--dataset_dir $DATASET_DIR \
				--dataset_name $DATASET_NAME \
				--output_dir $OUTPUT_DIR \
				--do_train \
				--do_predict \
				--seed $SEED \
				--remove_unused_columns False \
				--save_steps 100000 \
				--per_device_train_batch_size 8 \
				--per_device_eval_batch_size 32 \
				--num_train_epochs 10 \
				--optim "adamw_torch" \
				--learning_rate 1e-05 \
				--warmup_ratio 0.0 \
				--weight_decay 0.0 \
				--relation_representation "STANDARD_mention_pooling" \
				--use_entity_type_embeddings False \
				--overwrite_cache \
				--overwrite_output_dir

				: '
					--model_list bert-base-cased roberta-base allenai/biomed_roberta_base ~/BER-NLP/RE/RoBERTa-base-PM-M3-Voc-train-longer/RoBERTa-base-PM-M3-Voc-train-longer-hf \
								 bert-base-cased 
								 bert-large-cased
								 dmis-lab/biobert-base-cased-v1.1
								 dmis-lab/biobert-large-cased-v1.1 \
								 SpanBERT/spanbert-base-cased
								 SpanBERT/spanbert-large-cased
								 roberta-base
								 allenai/biomed_roberta_base
								 albert-base-v2 \ --> uncased model !!!
								 microsoft/deberta-base \
								 microsoft/deberta-v3-large \
								 microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \ --> uncased model !!!
								 microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \ --> uncased model !!!
								 allenai/scibert_scivocab_cased \
								 ~/BER-NLP/RE/models/KeBioLM \
								 ~/BER-NLP/RE/models/RoBERTa-base-PM-Voc/RoBERTa-base-PM-Voc-hf
								 ~/BER-NLP/RE/models/RoBERTa-base-PM-M3-Voc/RoBERTa-base-PM-M3-Voc-hf
								 ~/BER-NLP/RE/models/RoBERTa-large-PM-M3/RoBERTa-large-PM-M3-hf
								 ~/BER-NLP/RE/models/RoBERTa-large-PM-M3-Voc/RoBERTa-large-PM-M3-Voc-hf
								 ~/BER-NLP/RE/models/RoBERTa-base-PM-M3-Voc-train-longer/RoBERTa-base-PM-M3-Voc-train-longer-hf
								 ~/BER-NLP/RE/models/RoBERTa-base-PM-M3-Voc-distill/RoBERTa-base-PM-M3-Voc-distill-hf/
								 ~/BER-NLP/RE/models/RoBERTa-base-PM-M3-Voc-distill-align/RoBERTa-base-PM-M3-Voc-distill-align-hf
								 ~/BER-NLP/RE/models/biobert_v1.0_pmc

								 
					--config_name bert-base-cased \ --> used for the model dmis-lab/biobert-base-cased-v1.1
					--train_file $TRAIN_DATA \
					--validation_file $VALIDATION_DATA \
					--test_file $TEST_DATA \
					--do_train \
					--do_eval \
					--do_predict \
					--per_device_train_batch_size=16 \ --> 8 (default)
					--per_device_eval_batch_size=16 \ --> 8 (default)
					--optim="adamw_torch" \ --> "adamw_hf" (defalut) FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
					--learning_rate=5e-05 \ --> 5e-05 (default), 1e-05, 3e-05
					--warmup_ratio=0.1 \ --> 0.0 (default), 0.1 means 10%
					--weight_decay=0.01 \ --> 0.0 (default)
					--test_all_models \
					--do_lower_case=False \ --> False (default)
					--do_cross_validation \
					--num_of_folds=10 \
					--relation_representation EM_entity_start \ -->  STANDARD_cls_token, STANDARD_mention_pooling, \
																	EM_cls_token, EM_mention_pooling, EM_entity_start, \
																	POSITIONAL_mention_pooling_plus_context \
					--use_context "attn_based" \ --> None (default), "attn_based", "local"
					--use_entity_type_embeddings True \ -> False (default)
					--use_entity_typed_marker \ --> "Whether to use entity typed marker. E.g., [GENE], [/GENE] instead of [E1], [/E1] "
												    "This value is used in conjunction with EM representation."
					
					--metric_for_best_model "eval_f1" \
					--load_best_model_at_end True \
					--evaluation_strategy "epoch" \
					--save_strategy "epoch" \
					
					--seed $SEED \ --> 42 (default)
					--save_predictions \
				'
				
			break
            ;;
		"Tune Model (search hyperparameter)")
			echo "you chose Tune Model (search hyperparameter)."

			export DATASET_DIR=~/BER-NLP/RE/datasets
			export OUTPUT_DIR=/hpcgpfs01/scratch/gpark/RE_results
			
			#export DATASET_NAME=ChemProt_BLURB
			export DATASET_NAME=DDI_BLURB
			#export DATASET_NAME=GAD_BLURB
			
			
			#srun -p volta -A nlp-sbu -t 24:00:00 -N 1 --gres=gpu:2 -J re \
			#srun -p voltadebug -A nlp-sbu -t 4:00:00 -N 1 --gres=gpu:2 -J re \

			srun -p volta -A nlp-sbu -t 24:00:00 -N 1 --gres=gpu:2 -J re \
			python ~/BER-NLP/RE/re/optimize_model.py \
				--model_list microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
				--task_name re \
				--dataset_dir $DATASET_DIR \
				--dataset_name $DATASET_NAME \
				--output_dir $OUTPUT_DIR \
				--do_train \
				--do_eval \
				--evaluation_strategy="epoch" \
				--metric_for_best_model="eval_f1" \
				--optim="adamw_torch" \
				--relation_representation STANDARD_mention_pooling \
				--use_context attn_based \
				--remove_unused_columns=False \
				--save_steps=100000 \
				--overwrite_cache \
				--overwrite_output_dir
				
				
				: '
					--evaluation_strategy="epoch" \
					--metric_for_best_model="eval_f1"
				'
				
			break
			;;
			
		"Converting Tensorflow Checkpoints")
			export BERT_BASE_DIR=~/BER-NLP/RE/models/biobert_v1.0_pmc

			transformers-cli convert --model_type bert \
				--tf_checkpoint $BERT_BASE_DIR/bert_model.ckpt \
				--config $BERT_BASE_DIR/bert_config.json \
				--pytorch_dump_output $BERT_BASE_DIR/pytorch_model.bin
			
			
			break
			;;
		"Quit")
            break
            ;;
        *) echo "invalid option $REPLY";;
    esac
done