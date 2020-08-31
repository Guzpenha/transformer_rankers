export CUDA_VISIBLE_DEVICES=0,1,2,3
source /ssd/home/gustavo/transformer_rankers/env/bin/activate
REPO_DIR=/ssd/home/gustavo/transformer_rankers
ANSERINI_FOLDER=/ssd/home/gustavo/anserini/

#MC Dropout
for SEED in 42
do
    python ../examples/bert_ranker_train_trecpr2020_pred_clariq.py \
        --data_folder $REPO_DIR/data/ \
        --output_dir $REPO_DIR/data/output_data_clariq/ \
        --sample_data -1 \
        --max_seq_len 512 \
        --num_validation_instances 1000 \
        --validate_every_epochs 2 \
        --num_epochs 1 \
        --train_batch_size 6 \
        --val_batch_size 6 \
        --num_ns_train 1 \
        --num_ns_eval 9 \
        --seed ${SEED} \
        --anserini_folder $ANSERINI_FOLDER \
        --predict_with_uncertainty_estimation \
        --num_foward_prediction_passes 5
done

#Ensemble
for SEED in 42 1 2 3 4
do
    python ../examples/bert_ranker_train_trecpr2020_pred_clariq.py \
        --data_folder $REPO_DIR/data/ \
        --output_dir $REPO_DIR/data/output_data_clariq/ \
        --sample_data -1 \
        --max_seq_len 512 \
        --num_validation_instances 1000 \
        --validate_every_epochs 2 \
        --num_epochs 1 \
        --train_batch_size 6 \
        --val_batch_size 6 \
        --num_ns_train 1 \
        --num_ns_eval 9 \
        --seed ${SEED} \
        --anserini_folder $ANSERINI_FOLDER
done