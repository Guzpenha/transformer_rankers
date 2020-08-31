#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=12000
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24
source ~/env_slice_learning/bin/activate 
ANSERINI_FOLDER=/NOT_AVAILABLE/
REPO_DIR=/tudelft.net/staff-umbrella/conversationalsearch/transformer_rankers/

for TASK in 'mantis' 'msdialog' 'ubuntu_dstc8'
do
    for TRAIN_NEG_SAMPLER in 'bm25'
    do
        for TEST_NEG_SAMPLER in 'bm25' 'random' 'sentenceBERT'
        do
            for SEED in 42 1 2 3 4
            do
                srun python ../examples/crr_bert_ranker_example.py \
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
                    --num_ns_eval 9 \
                    --train_negative_sampler $TRAIN_NEG_SAMPLER \
                    --test_negative_sampler $TEST_NEG_SAMPLER \
                    --seed $SEED \
                    --anserini_folder $ANSERINI_FOLDER \
                    --bert_sentence_model $REPO_DIR/data/${TASK}/bert-base-cased_${TASK}
                    #--predict_with_uncertainty_estimation \
                    #--num_foward_prediction_passes 5 \
            done
        done
    done
done