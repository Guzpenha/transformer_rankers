export CUDA_VISIBLE_DEVICES=5,7
source /ssd/gustavo/transformer_rankers/env/bin/activate
REPO_DIR=/ssd/gustavo/transformer_rankers
ANSERINI_FOLDER=/ssd/gustavo/anserini/

for TASK in 'msdialog' 'mantis' 'ubuntu_dstc8'
do    
    for N_TRAIN_DOCS in 1 2 3 4 5 6 7 8 9
    do
        python ../examples/crr_bert_cross_task_cross_ns.py \
            --task $TASK \
            --data_folder $REPO_DIR/data/ \
            --output_dir $REPO_DIR/data/output_data_n_docs/ \
            --sample_data -1 \
            --max_seq_len 512 \
            --num_validation_instances 1000 \
            --validate_every_epochs 2 \
            --num_epochs 1 \
            --train_batch_size 6 \
            --val_batch_size 6 \
            --num_ns_train $N_TRAIN_DOCS \
            --num_ns_eval 9 \
            --seed 42 \
            --anserini_folder $ANSERINI_FOLDER \
            --bert_sentence_model $REPO_DIR/data/${TASK}/bert-base-cased_${TASK}
    done
done