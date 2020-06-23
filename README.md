# Transformer-Rankers
A library to conduct ranking experiments with transformers.


##Setup
Inside a python (3.6>) virtual enviroment run:

    pip install -e .

##Example
The following example uses BERT for the task of conversation response ranking using [MANtIS](https://guzpenha.github.io/MANtIS/) corpus, for more details of this approach refer to this [paper](https://arxiv.org/abs/1912.08555). See other examples, for instance how to use T5 for this task in [*transformer_ranker/examples/*](https://github.com/Guzpenha/transformer_rankers/tree/master/transformer_rankers/examples).

```python
from transformer_rankers.negative_samplers.negative_sampling import RandomNegativeSampler
from transformer_rankers.datasets.preprocess_crr import read_crr_tsv_as_df
from transformer_rankers.datasets.crr_dataset import CRRDataLoader
from transformers import BertTokenizer, BertForSequenceClassification

#Read dataset
data_folder = "data"
task = "mantis"
train = read_crr_tsv_as_df("{}/{}/train.tsv".format(data_folder, task))
valid = read_crr_tsv_as_df("{}/{}/valid.tsv".format(data_folder, task))

#Instantiate random negative samplers (1 for training 9 negative candidates for test)
ns_train = RandomNegativeSampler(list(train["response"].values), 1)
ns_val = RandomNegativeSampler(list(valid["response"].values) + \
    list(train["response"].values), 9)

#Create the loaders for the datasets, with the respective negative samplers
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
dataloader = CRRDataLoader(args=args, train_df=train,
                                val_df=valid, test_df=valid,
                                tokenizer=tokenizer, negative_sampler_train=ns_train,
                                negative_sampler_val=ns_val, task_type='classification')
train_loader, val_loader, test_loader = dataloader.get_pytorch_dataloaders()

#Use BERT to rank responses
model = BertForSequenceClassification.from_pretrained(args.transformer_model)
model.resize_token_embeddings(len(dataloader.tokenizer))

#Instantiate trainer that handles fitting.
trainer = TransformerTrainer(args, model, train_loader, val_loader, test_loader, 9, "classification", tokenizer)

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


##Main modules

### Datasets

Currently there is support for the following conversation response ranking datasets: [MANtIS](https://guzpenha.github.io/MANtIS/), [MSDialog](https://ciir.cs.umass.edu/downloads/msdialog/) and [Ubuntu v2](https://github.com/dstc8-track2/NOESIS-II/) from DSTC8. To automatically download the datasets use [scripts/download_crr_data.sh](https://github.com/Guzpenha/transformer_rankers/blob/master/transformer_rankers/scripts/download_crr_data.sh).

### Negative Samplers
Currently there is support to query for negative samples using the following approaches:
- **Random**: Selects a random document (response in conversation response ranking datasets).
- **BM25**: Uses [pyserini](https://github.com/castorini/pyserini/) to do the retrieval. Requires anserini installation.
- **sentenceBERT**: Uses [sentence embeddings](https://github.com/UKPLab/sentence-transformers) to calculate the embeddings of the query and candidates and [faiss]() to do fast retrieval of similar embeddings.

### Evaluation
- Uses trec_eval through [pytrec_eval](https://github.com/cvangysel/pytrec_eval) library to support most IR evaluation metrics.
- Additionaly R_X@K, i.e. recall with X candidates at K, was implemented.
- [Sacred](https://github.com/IDSIA/sacred) is used to log experiments, which can then be easily analyzed, c.f. [examples/crr_results_analyses_example.py](https://github.com/Guzpenha/transformer_rankers/blob/master/transformer_rankers/examples/crr_results_analyses_example.py)

### Trainer
Transformer trainer supports encoder-only transformers, e.g. BERT, and also encoder-decoder transformers, e.g. T5. An evaluation metric, such as nDCG, can be calculated during training using the validation set (using *--validate_epochs*).