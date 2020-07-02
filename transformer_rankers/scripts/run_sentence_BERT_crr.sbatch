#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=12000
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24
source ~/env_slice_learning/bin/activate 
REPO_DIR=/tudelft.net/staff-umbrella/conversationalsearch/transformer_rankers/

for TASK in 'mantis' 'msdialog' 'ubuntu_dstc8'
do    
    python train_sentenceBERT_crr.py \
        --task $TASK \
        --data_folder $REPO_DIR/data/ \
        --output_dir $REPO_DIR/data/$TASK/ \
        --num_epochs 1
done