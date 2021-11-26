export CUDA_VISIBLE_DEVICES=7
source /ssd/gustavo/transformer_rankers/env/bin/activate
REPO_DIR=/ssd/gustavo/transformer_rankers
ANSERINI_FOLDER=/ssd/gustavo/anserini/

BASE_MODEL='bert-base-cased'
# BASE_MODEL='all-mpnet-base-v2'

# for TASK in 'ubuntu_dstc8' 'mantis' 'msdialog'
# do
#     for NS in 'random' 'bm25' 'sentence_transformer'
#     do
#         python train_sentenceBERT_crr.py \
#             --task $TASK \
#             --data_folder $REPO_DIR/data/ \
#             --output_dir $REPO_DIR/data/output_data/ \
#             --train_batch_size 5 \
#             --anserini_folder $ANSERINI_FOLDER \
#             --negative_sampler $NS \
#             --transformer_model $BASE_MODEL \
#             --sentence_bert_ns_model 'all-mpnet-base-v2' \
#             --num_epochs 2 \
#             --loss 'MultipleNegativesRankingLoss'
#     done
# done

#pre-trained models
# for TASK in 'mantis' 'msdialog' 'ubuntu_dstc8'
# do
#     for SENT_BERT_MODEL in 'multi-qa-mpnet-base-dot-v1' 'msmarco-bert-base-dot-v5' 'all-mpnet-base-v2' 'msmarco-distilbert-base-tas-b' 'msmarco-roberta-base-ance-firstp'
#     do
#             python ../examples/negative_sampling.py \
#             --task ${TASK} \
#             --data_folder $REPO_DIR/data/ \
#             --output_dir $REPO_DIR/data/output_data \
#             --anserini_folder $ANSERINI_FOLDER \
#             --num_ns 10 \
#             --sentence_bert_model $SENT_BERT_MODEL 
#     done
# done

###Fine-tuned models using random NS
# for TASK in 'mantis' 'msdialog' 'ubuntu_dstc8'
# do
#         python ../examples/negative_sampling.py \
#         --task ${TASK} \
#         --data_folder $REPO_DIR/data/ \
#         --output_dir $REPO_DIR/data/output_data \
#         --anserini_folder $ANSERINI_FOLDER \
#         --num_ns 10 \
#         --sentence_bert_model $REPO_DIR/data/output_data/${BASE_MODEL}_${TASK}_ns_random_loss_MultipleNegativesRankingLoss
# done

###Fine-tuned models using random bm25
# for TASK in 'mantis' 'msdialog' 'ubuntu_dstc8'
# do
#         python ../examples/negative_sampling.py \
#         --task ${TASK} \
#         --data_folder $REPO_DIR/data/ \
#         --output_dir $REPO_DIR/data/output_data \
#         --anserini_folder $ANSERINI_FOLDER \
#         --num_ns 10 \
#         --sentence_bert_model $REPO_DIR/data/output_data/${BASE_MODEL}_${TASK}_ns_bm25_loss_MultipleNegativesRankingLoss
# done