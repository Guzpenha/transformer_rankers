export CUDA_VISIBLE_DEVICES=0,1,2,3
source /ssd/gustavo/transformer_rankers/env/bin/activate
REPO_DIR=/ssd/gustavo/transformer_rankers
ANSERINI_FOLDER=/ssd/gustavo/anserini/

VALIDATE_EVERY_X_STEPS=-1
TRAIN_INSTANCES=100000
WANDB_PROJECT='ls-vs-wsls-test'

####################
#    TREC PR       #
####################

for SEED in 1 2 3 4 5
do
    for SMOOTHING in 0.0 0.4
    do
        python ../examples/pointwise_bert_ranker.py \
            --task 'trec2020pr' \
            --data_folder $REPO_DIR/data/ \
            --output_dir $REPO_DIR/data/output_data_ls_vs_wsls/ \
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
            --wandb_project $WANDB_PROJECT \
            --loss_function 'ws-label-smoothing-cross-entropy' \
            --smoothing $SMOOTHING \
            --use_ls_ts \
            --num_instances_TSLA 50000
    done
done

for SEED in 1 2 3 4 5
do
    for SMOOTHING in 0.3
    do
        python ../examples/pointwise_bert_ranker.py \
            --task 'trec2020pr' \
            --data_folder $REPO_DIR/data/ \
            --output_dir $REPO_DIR/data/output_data_ls_vs_wsls/ \
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
            --wandb_project $WANDB_PROJECT \
            --loss_function 'label-smoothing-cross-entropy' \
            --smoothing $SMOOTHING \
            --use_ls_ts \
            --num_instances_TSLA 50000
    done
done


####################
#       QQP        #
####################

for SEED in 1 2 3 4 5
do
    for SMOOTHING in 0.0 0.2
    do
        python ../examples/pointwise_bert_ranker.py \
            --task 'qqp' \
            --data_folder $REPO_DIR/data/ \
            --output_dir $REPO_DIR/data/output_data_ls_vs_wsls/ \
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
            --wandb_project $WANDB_PROJECT \
            --loss_function 'ws-label-smoothing-cross-entropy' \
            --smoothing $SMOOTHING \
            --use_ls_ts \
            --num_instances_TSLA 50000
    done
done

for SEED in 1 2 3 4 5
do
    for SMOOTHING in 0.2
    do
        python ../examples/pointwise_bert_ranker.py \
            --task 'qqp' \
            --data_folder $REPO_DIR/data/ \
            --output_dir $REPO_DIR/data/output_data_ls_vs_wsls/ \
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
            --wandb_project $WANDB_PROJECT \
            --loss_function 'label-smoothing-cross-entropy' \
            --smoothing $SMOOTHING \
            --use_ls_ts \
            --num_instances_TSLA 50000
    done
done



####################
#       MANtIS     #
####################

for SEED in 1 2 3 4 5
do
    for SMOOTHING in 0.0 0.3
    do
        python ../examples/pointwise_bert_ranker.py \
            --task 'mantis' \
            --data_folder $REPO_DIR/data/ \
            --output_dir $REPO_DIR/data/output_data_ls_vs_wsls/ \
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
            --wandb_project $WANDB_PROJECT \
            --loss_function 'ws-label-smoothing-cross-entropy' \
            --smoothing $SMOOTHING \
            --use_ls_ts \
            --num_instances_TSLA 50000
    done
done

for SEED in 1 2 3 4 5
do
    for SMOOTHING in 0.1
    do
        python ../examples/pointwise_bert_ranker.py \
            --task 'mantis' \
            --data_folder $REPO_DIR/data/ \
            --output_dir $REPO_DIR/data/output_data_ls_vs_wsls/ \
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
            --wandb_project $WANDB_PROJECT \
            --loss_function 'label-smoothing-cross-entropy' \
            --smoothing $SMOOTHING \
            --use_ls_ts \
            --num_instances_TSLA 50000
    done
done