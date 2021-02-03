export CUDA_VISIBLE_DEVICES=3,4,5,6,7
source /ssd/gustavo/transformer_rankers/env/bin/activate
REPO_DIR=/ssd/gustavo/transformer_rankers
ANSERINI_FOLDER=/ssd/gustavo/anserini/

VALIDATE_EVERY_X_STEPS=100
TRAIN_INSTANCES=300000
WANDB_PROJECT='library-crr-bert-baseline'

for SEED in 1 2 3 4 5
do 
    for TASK in 'mantis' 'msdialog' 'ubuntu_dstc8'
    do
        python ../examples/pointwise_bert_ranker.py \
            --task $TASK \
            --data_folder $REPO_DIR/data/ \
            --output_dir $REPO_DIR/data/output_data/ \
            --sample_data -1 \
            --max_seq_len 512 \
            --num_validation_batches 500 \
            --validate_every_epochs -1 \
            --validate_every_steps $VALIDATE_EVERY_X_STEPS \
            --train_negative_sampler bm25 \
            --test_negative_sampler bm25 \
            --num_epochs 1 \
            --num_training_instances $TRAIN_INSTANCES \
            --train_batch_size 8 \
            --val_batch_size 8 \
            --num_ns_train 9 \
            --num_ns_eval 9 \
            --seed $SEED \
            --anserini_folder $ANSERINI_FOLDER \
            --wandb_project $WANDB_PROJECT        
    done
done
