export CUDA_VISIBLE_DEVICES=4,5,6,7
source /ssd/home/gustavo/transformer_rankers/env/bin/activate
REPO_DIR=/ssd/home/gustavo/transformer_rankers
ANSERINI_FOLDER=/ssd/home/gustavo/anserini/

for TASK in 'msdialog' 'mantis' 'ubuntu_dstc8'
do
    for TRAIN_NEG_SAMPLER in 'bm25' 'random' 'sentenceBERT'
    do
        for TEST_NEG_SAMPLER in 'bm25' 'random' 'sentenceBERT'
        do
            for SEED in 42
            do
                python ../examples/crr_bert_ranker_example.py \
                    --task $TASK \
                    --data_folder $REPO_DIR/data/ \
                    --output_dir $REPO_DIR/data/output_data_uncertainty/ \
                    --sample_data -1 \
                    --max_seq_len 512 \
                    --num_validation_instances 1000 \
                    --validate_every_epochs 2 \
                    --num_epochs 1 \
                    --train_batch_size 6 \
                    --val_batch_size 6 \
                    --num_ns_train 1 \
                    --num_ns_eval 19 \
                    --train_negative_sampler $TRAIN_NEG_SAMPLER \
                    --test_negative_sampler $TEST_NEG_SAMPLER \
                    --seed $SEED \
                    --anserini_folder $ANSERINI_FOLDER \
                    --bert_sentence_model $REPO_DIR/data/${TASK}/bert-base-cased_${TASK} \
                    --predict_with_uncertainty_estimation \
                    --num_foward_prediction_passes 5
            done
        done
    done
done