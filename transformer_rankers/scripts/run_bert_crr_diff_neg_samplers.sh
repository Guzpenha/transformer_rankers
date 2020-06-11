export CUDA_VISIBLE_DEVICES=4,5,6,7
source /ssd/home/gustavo/transformer_rankers/env/bin/activate
REPO_DIR=/ssd/home/gustavo/transformer_rankers

for TASK in 'mantis' 'msdialog' 'ubuntu_dstc8'
do
    for NEG_SAMPLER in 'random' 'tf-idf'
    do
        for SEED in 42 1 2 3 4
        do
            python ../examples/crr_bert_ranker_example.py \
                --task $TASK \
                --data_folder $REPO_DIR/data/ \
                --output_dir $REPO_DIR/data/output_data/ \
                --sample_data -1 \
                --max_seq_len 512 \
                --num_validation_instances 1000 \
                --validate_epochs 1 \
                --num_epochs 1 \
                --train_batch_size 8 \
                --val_batch_size 8 \
                --num_ns_train 1 \
                --num_ns_eval 19 \
                --negative_sampler $NEG_SAMPLER \
                --seed $SEED
        done
    done
done