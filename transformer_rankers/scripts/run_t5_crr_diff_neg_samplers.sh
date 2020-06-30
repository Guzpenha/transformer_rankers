export CUDA_VISIBLE_DEVICES=4,5,6,7
source /ssd/home/gustavo/transformer_rankers/env/bin/activate
REPO_DIR=/ssd/home/gustavo/transformer_rankers
ANSERINI_FOLDER=/ssd/home/gustavo/anserini/

for TASK in 'mantis' 'msdialog' 'ubuntu_dstc8'
do
    for TRAIN_NEG_SAMPLER in 'bm25' 'random' 'sentenceBERT'
    do
        for TEST_NEG_SAMPLER in 'bm25' 'random' 'sentenceBERT'
        do
            for SEED in 42 
            do
                python ../examples/crr_T5_ranker_example.py \
                    --task $TASK \
                    --data_folder $REPO_DIR/data/ \
                    --output_dir $REPO_DIR/data/output_data/ \
                    --sample_data -1 \
                    --max_seq_len 512 \
                    --num_validation_instances 1000 \
                    --validate_every_epochs 1 \
                    --num_epochs 1 \
                    --train_batch_size 8 \
                    --val_batch_size 8 \
                    --num_ns_train 1 \
                    --num_ns_eval 19 \
                    --train_negative_sampler $TRAIN_NEG_SAMPLER \
                    --test_negative_sampler $TEST_NEG_SAMPLER \
                    --seed $SEED \
                    --anserini_folder $ANSERINI_FOLDER \
                    --transformer_model 't5-base'
            done
        done
    done
done