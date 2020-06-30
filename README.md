# Transformer-Rankers
A library to conduct ranking experiments with transformers.


## Setup
Inside a python (>=3.6) virtual enviroment run:

    pip install -e .
    pip install -r requirements.txt

## Example: BERT-ranker for dialogue
The following example uses BERT for the task of conversation response ranking using [MANtIS](https://guzpenha.github.io/MANtIS/) corpus, for more details of this approach refer to this [paper](https://arxiv.org/abs/1912.08555). See other examples, for instance how to use T5 for this task in [*transformer_ranker/examples/*](https://github.com/Guzpenha/transformer_rankers/tree/master/transformer_rankers/examples).

```python
from transformer_rankers.trainers import transformer_trainer
from transformer_rankers.datasets import crr_dataset, preprocess_crr
from transformer_rankers.negative_samplers import negative_sampling 

#Read dataset
data_folder = "data"
task = "mantis"
train = preprocess_crr.read_crr_tsv_as_df("{}/{}/train.tsv".format(data_folder, task))
valid = preprocess_crr.read_crr_tsv_as_df("{}/{}/valid.tsv".format(data_folder, task))

#Instantiate random negative samplers (1 for training 9 negative candidates for test)
ns_train = negative_sampling.RandomNegativeSampler(list(train["response"].values), 1)
ns_val = negative_sampling.RandomNegativeSampler(list(valid["response"].values) + \
    list(train["response"].values), 9)

#Create the loaders for the datasets, with the respective negative samplers
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

dataloader = crr_dataset.CRRDataLoader(train_df=train, val_df=valid, test_df=valid,
                                tokenizer=tokenizer, negative_sampler_train=ns_train, 
                                negative_sampler_val=ns_val, task_type='classification', 
                                train_batch_size=32, val_batch_size=32, max_seq_len=512, 
                                sample_data=-1, "data/mantis")

train_loader, val_loader, test_loader = dataloader.get_pytorch_dataloaders()

#Use BERT to rank responses
model = BertForSequenceClassification.from_pretrained('bert-base-cased')
# we added [UTTERANCE_SEP] and [TURN_SEP] to the vocabulary so we need to resize the token embeddings
model.resize_token_embeddings(len(dataloader.tokenizer)) 

#Instantiate trainer that handles fitting.
trainer = transformer_trainer.TransformerTrainer(model=model,train_loader=train_loader,
                                val_loader=val_loader, test_loader=test_loader, 
                                num_ns_eval=9, task_type="classification", tokenizer=tokenizer,
                                validate_every_epoch=1, num_validation_instances=-1,
                                num_epochs=1, lr=0.0005, sacred_ex=None)

#Train the model
logging.info("Fitting BERT-ranker for MANtIS")
trainer.fit()

#Predict for test (in our example the validation set)
logging.info("Predicting")
preds = trainer.test()
```

The output of this script will look like this:

    [...]
    2020-06-23 11:19:44,522 [INFO] Epoch 1 val nDCG@10 
    2020-06-23 11:19:44,522 [INFO] Predicting
    2020-06-23 11:19:44,523 [INFO] Starting evaluation on test.
    2020-06-23 11:20:03,678 [INFO] Test recip_rank : 0.1345
    2020-06-23 11:20:03,678 [INFO] Test ndcg_cut_10 : 0.3236
    2020-06-23 11:20:03,746 [INFO] Result: 0.3236


## Main modules

### Datasets

Currently there is support for the following **conversation response ranking** datasets: [MANtIS](https://guzpenha.github.io/MANtIS/), [MSDialog](https://ciir.cs.umass.edu/downloads/msdialog/) and [Ubuntu v2](https://github.com/dstc8-track2/NOESIS-II/) from DSTC8. To automatically download the datasets use [scripts/download_crr_data.sh](https://github.com/Guzpenha/transformer_rankers/blob/master/transformer_rankers/scripts/download_crr_data.sh).

### Negative Samplers
Currently there is support to query for negative samples using the following approaches:
- **Random**: Selects a random document (response in conversation response ranking datasets).
- **BM25**: Uses [pyserini](https://github.com/castorini/pyserini/) to do the retrieval. Requires anserini installation, follow the *Getting Started* section of their [README](https://github.com/castorini/anserini).
- **sentenceBERT**: Pre-trained [sentence embeddings](https://github.com/UKPLab/sentence-transformers) to calculate the embeddings of the query and candidates, and [faiss](https://github.com/facebookresearch/faiss) is used to do fast retrieval of similar dense embeddings.

The documents retrieved by the query are checked against the relevant one, to avoid negatively sampling the correct candidate. The negative samplers can be used to do **full retrieval** over the documents list, due to their efficiency.

### Evaluation
- Uses trec_eval through [pytrec_eval](https://github.com/cvangysel/pytrec_eval) library to support most IR evaluation metrics, such as NDCG, MAP, MRR, etc. R_X@K, i.e. recall with X candidates at K, was implemented, since it is a common metric in conversation response ranking experiments.

- [Sacred](https://github.com/IDSIA/sacred) is used to log experiments, which receive unique IDs and store all hyperparameters that can then be used to analyze the results, c.f. [examples/crr_results_analyses_example.py](https://github.com/Guzpenha/transformer_rankers/blob/master/transformer_rankers/examples/crr_results_analyses_example.py).

### Trainers
Transformer trainer supports encoder-only transformers, e.g. BERT, and also encoder-decoder transformers, e.g. T5. An evaluation metric, such as nDCG, can be calculated during training using the validation set (using *--validate_epochs*). The metrics are stored on the sacred RUN_ID/metrics.json file.

### Models
Currently there is support for transformers for point-wise learning (similar to glue classification tasks) and also generative learning (predicting 'relevant' and 'not_relevant' tokens). For both approaches there is no need to change the huggingface transformers models, e.g. use directly *T5ForConditionalGeneration* or *BertForSequenceClassification*. The query (or conversation context) and document (or response) are concatenated and fed to the transformer model. The documents are then ordered by the logits predictions.

#### Uncertainty Estimation via Monte Carlo Dropout
Inspired by [*"Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning"*](https://arxiv.org/abs/1506.02142) *transformer-rankers* provides a function that does prediction with dropout at test time to get stochastic predictions of relevance. Such function provides a mean relevance probability and its respective uncertainty, i.e. variance, as opposed to the standard point estimates of relevance produced by deterministic rankers.

```python
average_logits, uncertainties = trainer.test_with_dropout(num_foward_prediction_passes=10)
```

## Experiments Results Examples
### Conversation response ranking
Validation set results, R<sub>10</sub>@1 values when using BM25 negative sampler (only 1 negative candidate for train) and finetunning for one epoch (BERT) and two epochs (T5). Use [*examples/crr_bert_ranker_example.py*](https://github.com/Guzpenha/transformer_rankers/blob/master/transformer_rankers/examples/crr_bert_ranker_example.py) and [*examples/crr_T5_ranker_example.py*](https://github.com/Guzpenha/transformer_rankers/blob/master/transformer_rankers/examples/crr_T5_ranker_example.py) to reproduce.

|             | MANtIS | MSDialog | Ubuntu v2 |
|-------------|--------|----------|-----------|
| BERT-ranker | 0.683  | 0.671    | 0.859     |
| T5-ranker |  0.616  |  0.650  |  0.826 |

<!-- ### Passage Retrieval

|             | ANTIQUE | MSMarco |
|-------------|--------|----------|
| BERT-ranker | -  |  -  |
| T5-ranker |  - |  - | -->
