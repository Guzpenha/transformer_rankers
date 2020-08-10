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
    git clone https://github.com/Guzpenha/transformer_rankers.git
    cd transformer_rankers    

    #Optionally use a virtual enviroment
    python3 -m venv env; source env/bin/activate    

    #Install the library and the requirements.
    pip install -e .
    pip install -r requirements.txt
```

## Example (I): BERT-ranker for retrieving similar questions

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

## Example (II): Different Transformer for a custom dataset
Check our documentation for an [example](https://guzpenha.github.io/transformer-rankers-doc/html/quick-start.html#example-ii-custom-dataset) with a custom dataset. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1h6N7uGMFWS5n5y95bUmxUdgPcVSU0xNu?usp=sharing)


## Example (III): BERT-ranker for clarifying questions

Check our colab for ranking clarifying questions to queries with BERT. The results are baselines for the [ClariQ](https://github.com/aliannejadi/ClariQ) challenge. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RHHbh5KQY-QDA7kV7wyHFJ7B_w5RRHzP?usp=sharing)


## News
10-08-2020: Transformer-rankers was used to generate baselines for the [ClariQ](https://github.com/aliannejadi/ClariQ) challenge.

10-07-2020: Get uncertainty estimates, i.e. variance, for rankers relevance predictions with [MC Dropout](https://arxiv.org/abs/1506.02142) at inference time using [*predict_with_uncertainty*](https://guzpenha.github.io/transformer-rankers-doc/html/_autosummary/transformer_rankers.trainers.transformer_trainer.TransformerTrainer.html#transformer_rankers.trainers.transformer_trainer.TransformerTrainer.predict_with_uncertainty).

09-07-2020: Transformer-rankers initial version realeased with support for 6 ranking datasets and negative sampling techniques (e.g. BM25, sentenceBERT similarity). The library uses [huggingface](https://huggingface.co/transformers/pretrained_models.html) pre-trained transformer models for ranking. See the main components at the documentation [page](https://guzpenha.github.io/transformer-rankers-doc/html/main-modules.html).

