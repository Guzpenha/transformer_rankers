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

#Create a virtual enviroment
python3 -m venv env; source env/bin/activate    

#Install the library and the requirements.
pip install -e .
pip install -r requirements.txt
```

## Colab Example
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jKTu8UMpG_eAe8RiPS0De4-FLRd49kRf?usp=sharing) Fine tune pointwise BERT for Community Question Answering.

## News
29-01-2021: Two papers recently used transformer-rankers library to conduct experiments: [On the Calibration and Uncertainty of Neural Learning to Rank Models for Conversational Search](https://arxiv.org/pdf/2101.04356.pdf) (EACL'21) and [Weakly Supervised Label Smoothing](https://arxiv.org/pdf/2012.08575.pdf) (ECIR'21).

15-09-2020: Cross Entropy [label smoothing](https://arxiv.org/pdf/1512.00567.pdf) was implemented as a loss function for learning to rank BERT models.

09-09-2020: Easily download and preprocess data for a task with [DataDownloader](https://github.com/Guzpenha/transformer_rankers/blob/master/transformer_rankers/examples/download_task_data.py). Currently 7 datasets for different retrieval tasks are implemented.

07-09-2020: [Pairwise BERT](https://github.com/Guzpenha/transformer_rankers/blob/master/transformer_rankers/models/pairwise_bert.py) ranker implemented.

10-08-2020: Transformer-rankers was used to generate baselines for the [ClariQ](https://github.com/aliannejadi/ClariQ) challenge.

10-07-2020: Get uncertainty estimates, i.e. variance, for rankers relevance predictions with [MC Dropout](https://arxiv.org/abs/1506.02142) at inference time using [*predict_with_uncertainty*](https://guzpenha.github.io/transformer-rankers-doc/html/_autosummary/transformer_rankers.trainers.transformer_trainer.TransformerTrainer.html#transformer_rankers.trainers.transformer_trainer.TransformerTrainer.predict_with_uncertainty).

09-07-2020: Transformer-rankers initial version realeased with support for 6 ranking datasets and negative sampling techniques (e.g. BM25, sentenceBERT similarity). The library uses [huggingface](https://huggingface.co/transformers/pretrained_models.html) pre-trained transformer models for ranking. See the main components at the documentation [page](https://guzpenha.github.io/transformer-rankers-doc/html/_autosummary/transformer_rankers.html).

