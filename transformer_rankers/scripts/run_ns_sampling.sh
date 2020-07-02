export CUDA_VISIBLE_DEVICES=1
source /ssd/home/gustavo/transformer_rankers/env/bin/activate

for TASK in 'mantis' 'msdialog' 'ubuntu_dstc8'
do
    python ../examples/negative_sampling_example.py \
        --task $TASK \
        --seed 1 \
        --data_folder /ssd/home/gustavo/transformer_rankers/data/ \
        --output_dir /ssd/home/gustavo/transformer_rankers/data/output_data/ \
        --sample_data -1 \
        --anserini_folder /ssd/home/gustavo/anserini/ \
        --num_ns_train 10

    python select_ns_examples.py --task $TASK \
        --output_dir /ssd/home/gustavo/transformer_rankers/data/output_data/
done