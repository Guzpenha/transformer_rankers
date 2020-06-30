export CUDA_VISIBLE_DEVICES=2,7
source /ssd/home/gustavo/transformer_rankers/env/bin/activate
REPO_DIR=/ssd/home/gustavo/transformer_rankers

for TASK in 'mantis' 'msdialog' 'ubuntu_dstc8'
do    
    python run_sentence_BERT.py \
        --task $TASK \
        --data_folder $REPO_DIR/data/ \
        --output_dir $REPO_DIR/data/$TASK/ \
        --num_epochs 1
done