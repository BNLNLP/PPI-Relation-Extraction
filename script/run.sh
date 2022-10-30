#!/bin/bash

PS3='Please enter your choice: '
options=("Run RE"
         "Quit")
select opt in "${options[@]}"
do
    case $opt in
        "Run RE")
            echo "you chose RE."

            export SEED=1

            export DATASET_DIR=~/BER-NLP/PPI-Relation-Extraction/datasets
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
            #export DATASET_NAME=PPI/type_annotation/Typed_PPI
            
            #srun -p volta -A nlp-sbu -t 24:00:00 -N 1 --gres=gpu:2 -J re \
            #srun -p voltadebug -A nlp-sbu -t 24:00:00 -N 1 --gres=gpu:2 -J re \

            srun -p volta -A nlp-sbu -t 24:00:00 -N 1 --gres=gpu:2 -J re \
            python ~/BER-NLP/PPI-Relation-Extraction/src/relation_extraction/run_re.py \
                --model_list dmis-lab/biobert-base-cased-v1.1 \
                --task_name "re" \
                --dataset_dir $DATASET_DIR \
                --dataset_name $DATASET_NAME \
                --output_dir $OUTPUT_DIR \
                --do_train \
                --do_predict \
                --seed $SEED \
                --remove_unused_columns False \
                --save_steps 100000 \
                --per_device_train_batch_size 16 \
                --per_device_eval_batch_size 32 \
                --num_train_epochs 10 \
                --optim "adamw_torch" \
                --learning_rate 5e-05 \
                --warmup_ratio 0.0 \
                --weight_decay 0.0 \
                --relation_representation "EM_entity_start" \
                --use_context "attn_based" \
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
                    --relation_representation EM_entity_start \ --> STANDARD_cls_token, STANDARD_mention_pooling, \
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
        "Quit")
            break
            ;;
        *) echo "invalid option $REPLY";;
    esac
done