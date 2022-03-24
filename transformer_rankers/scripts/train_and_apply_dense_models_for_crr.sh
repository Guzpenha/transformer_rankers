export CUDA_VISIBLE_DEVICES=0,1,2,3
source /ssd/gustavo/transformer_rankers/env/bin/activate
REPO_DIR=/ssd/gustavo/transformer_rankers
ANSERINI_FOLDER=/ssd/gustavo/anserini/

BASE_MODEL='all-mpnet-base-v2'
# BASE_MODEL='distilbert-base-cased' BASE_MODEL='bert-base-cased' BASE_MODEL='msmarco-bert-base-dot-v5' BASE_MODEL='microsoft/mpnet-base'

NUM_EPOCHS=1
BATCH_SIZE=5

LOSS='MultipleNegativesRankingLoss'
# Bad performing lossses: 'TripletLoss', 'ContrastiveLoss', 'MarginMSELoss', 'OnlineContrastiveLoss'

WANDB_PROJECT='full_rank_retrieval_dialogues'


# Dense models for the main table of the paper
for SEED in 42
    do
    for TASK in 'mantis' 'msdialog' 'ubuntu_dstc8'
    do
        for NS in 'random' 'bm25' 'sentence_transformer'
        do  
            python train_sentenceBERT_crr.py \
                --task $TASK \
                --data_folder $REPO_DIR/data/ \
                --output_dir $REPO_DIR/data/output_data/ \
                --train_batch_size $BATCH_SIZE \
                --anserini_folder $ANSERINI_FOLDER \
                --negative_sampler $NS \
                --transformer_model $BASE_MODEL \
                --sentence_bert_ns_model 'all-mpnet-base-v2' \
                --num_epochs $NUM_EPOCHS \
                --loss $LOSS \
                --seed $SEED \
                --wandb_project $WANDB_PROJECT
                # --external_corpus #Used for E5
                # --last_utterance_only  #Used for E4
                # --dont_remove_cand_subsets #Used for E3
                # --denoise_negatives #Used for E2
                # --num_ns_for_denoising 50 #Used for E2
        done
    done
done


##Using generative model as negative sampling (E6 of the paper)
# for GEN_MODEL in 'microsoft/DialoGPT-large' 'facebook/blenderbot-400M-distill'
#     do
#     for SEED in 42 #1 2 3 4
#         do
#         for TASK in 'mantis' 'msdialog' 'ubuntu_dstc8'
#            do
#             for NS in 'generative'
#             do  
#                 python train_sentenceBERT_crr.py \
#                     --task $TASK \
#                     --data_folder $REPO_DIR/data/ \
#                     --output_dir $REPO_DIR/data/output_data/ \
#                     --train_batch_size $BATCH_SIZE \
#                     --anserini_folder $ANSERINI_FOLDER \
#                     --negative_sampler $NS \
#                     --transformer_model $BASE_MODEL \
#                     --sentence_bert_ns_model 'all-mpnet-base-v2' \
#                     --num_epochs $NUM_EPOCHS \
#                     --loss $LOSS \
#                     --seed $SEED \
#                     --wandb_project $WANDB_PROJECT \
#                     --generative_sampling_model $GEN_MODEL
#             done
#         done
#     done
# done


## Using resp2context as NS (this is not in the paper.)
# for TASK in 'msdialog' #'mantis' 'ubuntu_dstc8'
# for TASK in $SINGLE_TASK
# do
#     python train_sentenceBERT_crr.py \
#         --task ${TASK}_resp2context_last_utt_True \
#         --data_folder $REPO_DIR/data/ \
#         --output_dir $REPO_DIR/data/output_data/ \
#         --train_batch_size $BATCH_SIZE \
#         --anserini_folder $ANSERINI_FOLDER \
#         --negative_sampler 'bm25' \
#         --transformer_model $BASE_MODEL \
#         --sentence_bert_ns_model 'all-mpnet-base-v2' \
#         --num_epochs $NUM_EPOCHS \
#         --loss $LOSS
# done