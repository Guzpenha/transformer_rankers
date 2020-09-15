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

## Colab Examples
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1h6N7uGMFWS5n5y95bUmxUdgPcVSU0xNu?usp=sharing) Using transformers for learning to rank from a pandas DF.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RHHbh5KQY-QDA7kV7wyHFJ7B_w5RRHzP?usp=sharing) Learning to rank clarifying questions with BERT-ranker.

## News
15-09-2020: Cross Entropy [label smoothing](https://arxiv.org/pdf/1512.00567.pdf) was implemented as a loss function for learning to rank BERT models.

09-09-2020: Easily download and preprocess data for a task with [DataDownloader](https://github.com/Guzpenha/transformer_rankers/blob/master/transformer_rankers/examples/download_task_data.py). Currently 7 datasets for different retrieval tasks are implemented.

07-09-2020: [Pairwise BERT](https://github.com/Guzpenha/transformer_rankers/blob/master/transformer_rankers/models/pairwise_bert.py) ranker implemented. Also updated huggingface transformers to 3.1.

10-08-2020: Transformer-rankers was used to generate baselines for the [ClariQ](https://github.com/aliannejadi/ClariQ) challenge.

10-07-2020: Get uncertainty estimates, i.e. variance, for rankers relevance predictions with [MC Dropout](https://arxiv.org/abs/1506.02142) at inference time using [*predict_with_uncertainty*](https://guzpenha.github.io/transformer-rankers-doc/html/_autosummary/transformer_rankers.trainers.transformer_trainer.TransformerTrainer.html#transformer_rankers.trainers.transformer_trainer.TransformerTrainer.predict_with_uncertainty).

09-07-2020: Transformer-rankers initial version realeased with support for 6 ranking datasets and negative sampling techniques (e.g. BM25, sentenceBERT similarity). The library uses [huggingface](https://huggingface.co/transformers/pretrained_models.html) pre-trained transformer models for ranking. See the main components at the documentation [page](https://guzpenha.github.io/transformer-rankers-doc/html/main-modules.html).

