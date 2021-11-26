export CUDA_VISIBLE_DEVICES=2
source /ssd/gustavo/transformer_rankers/env/bin/activate
REPO_DIR=/ssd/gustavo/transformer_rankers
ANSERINI_FOLDER=/ssd/gustavo/anserini/

# Training models with whole context and with last_utterance only
# python train_response2context.py  --data_folder $REPO_DIR/data/
# python train_response2context.py  --data_folder $REPO_DIR/data/ --last_utterance_only

# Applying the trained models
# for TASK in 'ubuntu_dstc8' 'mantis' #'msdialog'
# do    
#     python apply_response2context.py \
#         --data_folder $REPO_DIR/data/ \
#         --t5_model $REPO_DIR/data/t5-base_response2context/ \
#         --task $TASK

#     python apply_response2context.py \
#         --data_folder $REPO_DIR/data/ \
#         --t5_model $REPO_DIR/data/t5-base_response2context_last_utt_only/ \
#         --task $TASK
# done

# Evaluating the trained models using pyserini
for TASK in 'mantis' 'ubuntu_dstc8' 'msdialog'
do 
    ###baseline bm25
    python ../examples/negative_sampling.py \
        --task ${TASK} \
        --data_folder $REPO_DIR/data/ \
        --output_dir $REPO_DIR/data/output_data \
        --anserini_folder $ANSERINI_FOLDER \
        --num_ns 10

    ###bm25+resp2context 
    # python ../examples/negative_sampling.py \
    #     --task ${TASK}_resp2context_last_utt_False \
    #     --data_folder $REPO_DIR/data/ \
    #     --output_dir $REPO_DIR/data/output_data \
    #     --anserini_folder $ANSERINI_FOLDER \
    #     --num_ns 10

    ##bm25+resp2context_last_utt 
    # python ../examples/negative_sampling.py \
    #     --task ${TASK}_resp2context_last_utt_True \
    #     --data_folder $REPO_DIR/data/ \
    #     --output_dir $REPO_DIR/data/output_data \
    #     --anserini_folder $ANSERINI_FOLDER \
    #     --num_ns 10
done

# Evaluating the trained models using pyterrier
# for TASK in 'msdialog' 'mantis' 'ubuntu_dstc8'
# do 
    # baseline bm25
    # python ../examples/negative_sampling_pyterrier.py \
    #     --task ${TASK} \
    #     --data_folder $REPO_DIR/data/ \
    #     --output_dir $REPO_DIR/data/output_data \
    #     --num_ns 10

    ##bm25+resp2context 
    # python ../examples/negative_sampling_pyterrier.py \
    #     --task ${TASK}_resp2context_last_utt_False \
    #     --data_folder $REPO_DIR/data/ \
    #     --output_dir $REPO_DIR/data/output_data \
    #     --num_ns 10

    ##bm25+resp2context_last_utt 
    # python ../examples/negative_sampling_pyterrier.py \
    #     --task ${TASK}_resp2context_last_utt_True \
    #     --data_folder $REPO_DIR/data/ \
    #     --output_dir $REPO_DIR/data/output_data \
    #     --num_ns 10
# done