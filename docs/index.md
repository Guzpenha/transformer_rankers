---
layout: home
comments: false
seotitle: Transformer-Rankers - A library to conduct experiments with transformer-based rankers
description: 
---

A [library](https://github.com/Guzpenha/transformer_rankers) to conduct ranking experiments with pre-trained transformers.


## Setup
```bash
    #Clone the repo
    git clone git@github.com:Guzpenha/transformer_rankers.git
    cd transformer_rankers    

    #Optionally use a virtual enviroment
    python3 -m venv env; source env/bin/activate    

    #Install the library and the requirements.
    pip install -e .
    pip install -r requirements.txt
```

## Example: BERT-ranker for retrieving similar questions

The task is to rank similar questions to an input question. We download the data and train BERT-ranker (with only 1000 samples to be fast) using one of our example scripts:

```bash
    cd transformer_rankers/scripts
    ./download_sqr_data.sh

    python ../examples/crr_bert_ranker_example.py \
        --task qqp \
        --data_folder ../../data/ \
        --output_dir ../../data/output_data \
        --sample_data 1000
```

## News

10-07-2020: Get uncertainty estimates, i.e. variance, for rankers relevance predictions with [MC Dropout](https://arxiv.org/abs/1506.02142) at inference time using [*predict_with_uncertainty*](https://guzpenha.github.io/transformer-rankers-doc/html/_autosummary/transformer_rankers.trainers.transformer_trainer.TransformerTrainer.html#transformer_rankers.trainers.transformer_trainer.TransformerTrainer.predict_with_uncertainty).

09-07-2020: Transformer-rankers initial version realeased with support for 6 ranking datasets and negative sampling techniques (e.g. BM25, sentenceBERT similarity). The library uses [huggingface](https://huggingface.co/transformers/pretrained_models.html) pre-trained transformer models for ranking. See the main components at the documentation [page](https://guzpenha.github.io/transformer-rankers-doc/html/main-modules.html).


<!-- 
The output will be something like this:
```
   [...]
   2020-06-23 11:19:44,522 [INFO] Epoch 1 val nDCG@10: 0.245
   2020-06-23 11:19:44,522 [INFO] Predicting
   2020-06-23 11:19:44,523 [INFO] Starting evaluation on test.
   2020-06-23 11:20:03,678 [INFO] Test ndcg_cut_10: 0.3236
```

The experiment info will be saved at *../data/output_data*, where you can find the following files:
```bash
   /data/output_data/1/config.json
   /data/output_data/1/cout.txt
   /data/output_data/1/labels.csv
   /data/output_data/1/predictions.csv
   /data/output_data/1/run.json
```
You can easily aggregate the results of different experiment runs using */examples/crr_results_analyses_example.py*: -->



