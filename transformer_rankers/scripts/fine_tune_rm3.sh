source /ssd/gustavo/transformer_rankers/env/bin/activate
REPO_DIR=/ssd/gustavo/transformer_rankers
ANSERINI_FOLDER=/ssd/gustavo/anserini/

for TASK in 'mantis' 'msdialog' 'ubuntu_dstc8'
do
    for EXPANSION_TERMS in 5 10 15
    do
        for EXPANSION_DOCS in 5 10 15
        do
            for ORIGINAL_Q_WEIGHT in 0.5 0.7
            do
            python ../examples/rm3_hyperparameters.py \
            --task ${TASK} \
            --data_folder $REPO_DIR/data/ \
            --output_dir $REPO_DIR/data/output_data \
            --anserini_folder $ANSERINI_FOLDER \
            --num_ns 10 \
            --num_expansion_terms $EXPANSION_TERMS \
            --num_expansion_docs $EXPANSION_DOCS \
            --original_query_weight $ORIGINAL_Q_WEIGHT
            done
        done
    done
done