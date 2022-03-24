source /ssd/gustavo/transformer_rankers/env/bin/activate
REPO_DIR=/ssd/gustavo/transformer_rankers
ANSERINI_FOLDER=/ssd/gustavo/anserini/

for TASK in 'msdialog' 'mantis' 'ubuntu_dstc8'
do
    python ../examples/eval_random_rm3.py \
    --task ${TASK} \
    --data_folder $REPO_DIR/data/ \
    --output_dir $REPO_DIR/data/output_data \
    --anserini_folder $ANSERINI_FOLDER \
    --num_ns 10 
done