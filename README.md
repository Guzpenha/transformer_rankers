<img src="https://guzpenha.github.io/transformer_rankers/images/tRankers.png" align="right" height="90px"/>


# Transformer-Rankers
<a href="https://guzpenha.github.io/transformer-rankers-doc/html/index.html">
<img alt="Documentation" src="https://img.shields.io/badge/docs-latest-success.svg">
</a>
<a href="https://github.com/Guzpenha/transformer_rankers/blob/master/LICENSE">
<img alt="license" src="https://img.shields.io/badge/License-MIT-blue.svg">
</a>

Transformer-rankers is a library to conduct ranking experiments with transformers. 

Most of the research experiments performed focused on the task of conversation response ranking, see [EACL'21](https://arxiv.org/abs/2012.08575) and [ECIR'20](https://arxiv.org/abs/2101.04356). This repo is intended to be used to perform research experiments and not to create production ready systems. Better alternatives for general ranking models are either [pyterrier](https://pyterrier.readthedocs.io/en/latest/) or [pyserini](https://github.com/castorini/pyserini).

## Examples
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wGmaO3emC7Sg-tA7nGehIQ2vjOLN9S5e?usp=sharing) Fine tune pointwise BERT for conversation response ranking.

[![Wandb report](https://img.shields.io/badge/wandb-Open%20report-yellow) ](https://wandb.ai/guz/library-crr-bert-baseline/reports/BERT-ranker-baselines-for-CRR--Vmlldzo0NDcyMzU) Wandb report of fine tunning BERT for conversation response ranking.

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
## Code example: BERT-ranker for dialogue
The folowing example uses BERT for the task of conversation response ranking using [MANtIS](https://guzpenha.github.io/MANtIS/) corpus. We can download the data as follows:

```python
from transformer_rankers.datasets import downloader

#Download the data with DataDownloader
data_folder = "data"
dataDownloader = downloader.DataDownloader("mantis", data_folder)
dataDownloader.download_and_preprocess()
```
And train BERT for pointwise learning to rank with randomly sampled negative samples:
```python
from transformers import BertTokenizer
from transformer_rankers.models import pointwise_bert
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
                                train_batch_size=6, val_batch_size=6, max_seq_len=512, 
                                sample_data=-1, cache_path="{}/{}".format(data_folder, task))

train_loader, val_loader, test_loader = dataloader.get_pytorch_dataloaders()


model = pointwise_bert.BertForPointwiseLearning.from_pretrained('bert-base-cased')
# we added [UTTERANCE_SEP] and [TURN_SEP] to the vocabulary so we need to resize the token embeddings
model.resize_token_embeddings(len(dataloader.tokenizer)) 

#Instantiate trainer that handles fitting.
trainer = transformer_trainer.TransformerTrainer(model=model,train_loader=train_loader,
                                val_loader=val_loader, test_loader=test_loader, 
                                num_ns_eval=9, task_type="classification", tokenizer=tokenizer,
                                validate_every_epoch=1, num_validation_batches=-1,
                                num_epochs=1, lr=0.0005, sacred_ex=None,
                                validate_every_steps=-1, num_training_instances=-1)

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