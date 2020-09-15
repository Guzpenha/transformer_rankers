<img src="https://guzpenha.github.io/transformer_rankers/images/tRankers.png" align="right" height="90px"/>


# Transformer-Rankers
<a href="https://guzpenha.github.io/transformer-rankers-doc/html/index.html">
<img alt="Documentation" src="https://img.shields.io/badge/docs-latest-success.svg">
</a>
<a href="https://github.com/Guzpenha/transformer_rankers/blob/master/LICENSE">
<img alt="license" src="https://img.shields.io/badge/License-MIT-blue.svg">
</a>

A library to conduct ranking experiments with transformers. 


## Setup
The following will clone the repo, install a virtual env and install the library with the requirements.
```bash
#Clone the repo
git clone https://github.com/Guzpenha/transformer_rankers.git
cd transformer_rankers    

#Optionally use a virtual enviroment
python3 -m venv env
source env/bin/activate

#Optionally use a virtual enviroment
pip install -e .
pip install -r requirements.txt
```
## Example: BERT-ranker for dialogue
The following example uses BERT for the task of conversation response ranking using [MANtIS](https://guzpenha.github.io/MANtIS/) corpus. We can download the data as follows:

```python
from transformer_rankers.datasets import downloader

#Download the data with DataDownloader
data_folder = "data"
dataDownloader = downloader.DataDownloader("mantis", data_folder)
dataDownloader.download_and_preprocess()
```
And train BERT for pointwise learning to rank with randomly sampled negative samples:
```python
from transformer_rankers.trainers import transformer_trainer
from transformer_rankers.datasets import dataset, preprocess_crr
from transformer_rankers.negative_samplers import negative_sampling 
from transformer_rankers.eval import results_analyses_tools

#Load the dataset
task = "mantis"
train = pd.read_csv(data_folder+task+"/train.tsv", sep="\t")
valid = pd.read_csv(data_folder+task+"/valid.tsv", sep="\t")

#Instantiate random negative samplers (1 for training 9 negative candidates for test)
# the library also supports BM25 and sentenceBERT negative samplers.
ns_train = negative_sampling.RandomNegativeSampler(list(train["response"].values), 1)
ns_val = negative_sampling.RandomNegativeSampler(list(valid["response"].values) + \
    list(train["response"].values), 9)

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
special_tokens_dict = {'additional_special_tokens': ['[UTTERANCE_SEP]', '[TURN_SEP]'] }
tokenizer.add_special_tokens(special_tokens_dict)

#Create the loaders for the datasets, with the respective negative samplers        
dataloader = dataset.QueryDocumentDataLoader(train_df=train, val_df=valid, test_df=valid,
                                tokenizer=tokenizer, negative_sampler_train=ns_train, 
                                negative_sampler_val=ns_val, task_type='classification', 
                                train_batch_size=32, val_batch_size=32, max_seq_len=512, 
                                sample_data=-1, "data/mantis")

train_loader, val_loader, test_loader = dataloader.get_pytorch_dataloaders()


model = BertForSequenceClassification.from_pretrained('bert-base-cased')
# we added [UTTERANCE_SEP] and [TURN_SEP] to the vocabulary so we need to resize the token embeddings
model.resize_token_embeddings(len(dataloader.tokenizer)) 

#Instantiate trainer that handles fitting.
trainer = transformer_trainer.TransformerTrainer(model=model,train_loader=train_loader,
                                val_loader=val_loader, test_loader=test_loader, 
                                num_ns_eval=9, task_type="classification", tokenizer=tokenizer,
                                validate_every_epoch=1, num_validation_batches=-1,
                                num_epochs=1, lr=0.0005, sacred_ex=None)

#Train the model
logging.info("Fitting BERT-ranker for MANtIS")
trainer.fit()

#Predict for test (in our example the validation set)
logging.info("Predicting")
preds, labels, _ = trainer.test()
res = results_analyses_tools.\
    evaluate_and_aggregate(preds, labels, ['ndcg_cut_10'])

for metric, v in res.items():
    logging.info("Test {} : {:4f}".format(metric, v))
```

The output will look like this:

    [...]
    2020-06-23 11:19:44,522 [INFO] Epoch 1 val nDCG@10: 0.245
    2020-06-23 11:19:44,522 [INFO] Predicting
    2020-06-23 11:19:44,523 [INFO] Starting evaluation on test.
    2020-06-23 11:20:03,678 [INFO] Test ndcg_cut_10: 0.3236

If you login to [wandb](https://app.wandb.ai/home), the trainer will handle logging loss and validation metrics to it.

## Colab examples
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1h6N7uGMFWS5n5y95bUmxUdgPcVSU0xNu?usp=sharing) Fine-tune different transformers for a dataset in pandas DataFrame format. 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RHHbh5KQY-QDA7kV7wyHFJ7B_w5RRHzP?usp=sharing) Fine-tune BERT-ranker for ranking clarifying questions to queries (RQ2 of ClariQ challenge).


## Code Organization

### transformer_rankers/datasets

Stores processors for specific datasets as well as code to generate pytorch datasets To download the datasets use *scripts/download_\<task>_data.sh*. Currently implemented processors: 

- **conversation response ranking**: [MANtIS](https://guzpenha.github.io/MANtIS/), [MSDialog](https://ciir.cs.umass.edu/downloads/msdialog/) and [Ubuntu v2](https://github.com/dstc8-track2/NOESIS-II/) from DSTC8.
- **similar question retrieval**: [Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs) and [LinkSO](https://sites.google.com/view/linkso)
- **passage retrieval**: [TREC 2020 Passage Ranking](https://microsoft.github.io/TREC-2020-Deep-Learning/).
- **clarifying question retrieval**: [ClariQ](https://github.com/aliannejadi/ClariQ).

Note that since we choose the negative sampling on the go, we do not read the negative samples from the datasets, only the relevant query-document combinations. Example extracted from TREC 2020 Passage Retrieval, in the format expected for *QueryDocumentDataset*:

| Query | Relevant Document |
|-------------|--------|
| why do my eyes water | Watering eyes occur if too many tears are produced [...] |
| why do many substances dissolve in water, but others do not? | Quick Answer. Substances that have ionic molecules [...]| 

### transformer_rankers/negative_samplers
Currently there is support to query for negative samples using the following approaches:
- **Random**: Selects a random document.
- **BM25**: Uses [pyserini](https://github.com/castorini/pyserini/) to do the retrieval with BM25. Requires anserini installation, follow the *Getting Started* section of their [README](https://github.com/castorini/anserini).
- **sentenceBERT**: Uses [sentence embeddings](https://github.com/UKPLab/sentence-transformers) to calculate dense representations of the query and candidates, and [faiss](https://github.com/facebookresearch/faiss) is used to do fast retrieval, i.e. dense similarity computation.

See [negative_sampling.py](https://github.com/Guzpenha/transformer_rankers/blob/master/transformer_rankers/examples/negative_sampling.py) for an example of the negative samplers.

### transformer_rankers/eval
Uses trec_eval through [pytrec_eval](https://github.com/cvangysel/pytrec_eval) library to support most IR evaluation metrics, such as NDCG, MAP, MRR, etc. Additional metrics are implemented here, such as Recall_with_n_candidates@K.

### transformer_rankers/trainers
Transformer trainer supports encoder-only transformers, e.g. BERT, and also encoder-decoder transformers, e.g. T5, from the huggingface transformers library, see their pre-trained [models](https://huggingface.co/transformers/pretrained_models.html).

### transformer_rankers/models
This is the module where you can find different neural ranker implementations. Currently there is suppport for pairwise learning using BERT. See [pairwise_bert_ranker.py](https://github.com/Guzpenha/transformer_rankers/blob/master/transformer_rankers/examples/pairwise_bert_ranker.py).

## Experimental Results Examples 
All results consider the problem of re-ranking from a list of **9 negative samples (using BM25) and the relevant document**.

### Passage Retrieval (nDCG@10)

|             | TREC-DL-PR 2020|
|-------------|--------|
| BERT-ranker | 0.715  |


### Conversation response ranking (R<sub>10</sub>@1)

|             | MANtIS | MSDialog | Ubuntu DSTC8-task1 |
|-------------|--------|----------|-----------|
| BERT-ranker | 0.683  | 0.671    | 0.859     |
| T5-ranker |  0.616  |  0.650  |  0.826 |

### Similar question ranking (MAP)

|             | Quora | LinkSO |
|-------------|--------|----------|
| BERT-ranker |  0.536 | 0.658 |
| T5-ranker |  0.578  |  0.389 |