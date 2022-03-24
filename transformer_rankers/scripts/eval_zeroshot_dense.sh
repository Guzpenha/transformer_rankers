export CUDA_VISIBLE_DEVICES=2
source /ssd/gustavo/transformer_rankers/env/bin/activate
REPO_DIR=/ssd/gustavo/transformer_rankers
ANSERINI_FOLDER=/ssd/gustavo/anserini/

## EVALUATION of the models full rank effectiveness\
#pre-trained models
for TASK in 'msdialog' 'mantis' 'ubuntu_dstc8'
do
    for SENT_BERT_MODEL in 'msmarco-roberta-base-ance-firstp' 'msmarco-distilbert-base-tas-b' 'msmarco-bert-base-dot-v5' 'multi-qa-mpnet-base-dot-v1' 'all-mpnet-base-v2' 
    do
            python ../examples/eval_dense_models.py \
            --task ${TASK} \
            --data_folder $REPO_DIR/data/ \
            --output_dir $REPO_DIR/data/output_data \
            --anserini_folder $ANSERINI_FOLDER \
            --num_ns 10 \
            --sentence_bert_model $SENT_BERT_MODEL 
    done
done