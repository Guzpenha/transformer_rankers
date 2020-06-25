#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=4000
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24
source /tudelft.net/staff-umbrella/conversationalsearch/env_transformer_rankers/bin/activate
ANSERINI_FOLDER=/NOT_AVAILABLE/
REPO_DIR=/ssd/home/gustavo/transformer_rankers

for TASK in 'mantis' 'msdialog' 'ubuntu_dstc8'
do
    for TRAIN_NEG_SAMPLER in 'bm25' 'random' 'sentenceBERT'
    do
        for TEST_NEG_SAMPLER in 'bm25' 'random' 'sentenceBERT'
        do
            for SEED in 42 1 2 3 4
            do
                srun python ../examples/crr_bert_ranker_example.py \
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
                    --train_negative_sampler $TRAIN_NEG_SAMPLER \
                    --test_negative_sampler $TEST_NEG_SAMPLER \
                    --seed $SEED \
                    --anserini_folder $ANSERINI_FOLDER
            done
        done
    done
done

for TASK in 'mantis' 'msdialog' 'ubuntu_dstc8'
do
    for TRAIN_NEG_SAMPLER in 'bm25' 'random' 'sentenceBERT'
    do
        for TEST_NEG_SAMPLER in 'bm25' 'random' 'sentenceBERT'
        do
            for SEED in 42 1 2 3 4
            do
                srun python ../examples/crr_T5_ranker_example.py \
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
                    --train_negative_sampler $TRAIN_NEG_SAMPLER \
                    --test_negative_sampler $TEST_NEG_SAMPLER \
                    --seed $SEED \
                    --anserini_folder $ANSERINI_FOLDER \
                    --transformer_model 't5-base'
            done
        done
    done
done