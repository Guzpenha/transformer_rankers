# Transformer-Rankers
A library to conduct ranking experiments with transformers.


## Setup
Inside a python (>=3.6) virtual enviroment run:

    pip install -e .
    pip install -r requirements.txt

## Example: BERT-ranker for dialogue
The following example uses BERT for the task of conversation response ranking using [MANtIS](https://guzpenha.github.io/MANtIS/) corpus. See other examples, for instance how to use T5 for this task in [*transformer_ranker/examples/*](https://github.com/Guzpenha/transformer_rankers/tree/master/transformer_rankers/examples).

```python
from transformer_rankers.trainers import transformer_trainer
from transformer_rankers.datasets import dataset, preprocess_crr
from transformer_rankers.negative_samplers import negative_sampling 
from transformer_rankers.eval import results_analyses_tools

#Read dataset
data_folder = "data"
task = "mantis"
train = preprocess_crr.read_crr_tsv_as_df("{}/{}/train.tsv".format(data_folder, task))
valid = preprocess_crr.read_crr_tsv_as_df("{}/{}/valid.tsv".format(data_folder, task))

#Instantiate random negative samplers (1 for training 9 negative candidates for test)
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
preds, labels = trainer.test()
res = results_analyses_tools.\
    evaluate_and_aggregate(preds, labels, ['ndcg_cut_10'])

for metric, v in res.items():
    logging.info("Test {} : {:4f}".format(metric, v))
```

The output will look like this:

    [...]
    2020-06-23 11:19:44,522 [INFO] Epoch 1 val nDCG@10 
    2020-06-23 11:19:44,522 [INFO] Predicting
    2020-06-23 11:19:44,523 [INFO] Starting evaluation on test.
    2020-06-23 11:20:03,678 [INFO] Test ndcg_cut_10 : 0.3236


## Code Organization

### datasets

Stores processors for specific datasets as well as code to generate pytorch datasets To download the datasets use *scripts/download_\<task>_data.sh*. Implemented processors: 
- **conversation response ranking**: [MANtIS](https://guzpenha.github.io/MANtIS/), [MSDialog](https://ciir.cs.umass.edu/downloads/msdialog/) and [Ubuntu v2](https://github.com/dstc8-track2/NOESIS-II/) from DSTC8.
- **similar question retrieval**: [Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs) and [LinkSO](https://sites.google.com/view/linkso)

### negative_samplers
Currently there is support to query for negative samples using the following approaches:
- **Random**: Selects a random document.
- **BM25**: Uses [pyserini](https://github.com/castorini/pyserini/) to do the retrieval with BM25. Requires anserini installation, follow the *Getting Started* section of their [README](https://github.com/castorini/anserini).
- **sentenceBERT**: Uses [sentence embeddings](https://github.com/UKPLab/sentence-transformers) to calculate dense representations of the query and candidates, and [faiss](https://github.com/facebookresearch/faiss) is used to do fast retrieval, i.e. dense similarity computation.

See [negative_sampling_example.py](https://github.com/Guzpenha/transformer_rankers/blob/master/transformer_rankers/examples/negative_sampling_example.py) for an example of using the negative samplers.


### examples
Examples of using the library such as  to train transformer-based rankers and evaluate the results.

### eval
Uses trec_eval through [pytrec_eval](https://github.com/cvangysel/pytrec_eval) library to support most IR evaluation metrics, such as NDCG, MAP, MRR, etc. Additional metrics are implemented here, such as Recall_with_n_candidates@K.


### trainers
Transformer trainer supports encoder-only transformers, e.g. BERT, and also encoder-decoder transformers, e.g. T5, from the huggingface transformers library, see their pre-trained [models](https://huggingface.co/transformers/pretrained_models.html).

<!-- ### Models -->
<!-- Currently there is support for transformers for point-wise learning (similar to glue classification tasks) and also generative learning (predicting 'relevant' and 'not_relevant' tokens). For both approaches there is no need to change the huggingface transformers models, e.g. use directly *T5ForConditionalGeneration* or *BertForSequenceClassification*. The query (or conversation context) and document (or response) are concatenated and fed to the transformer model. The documents are then ordered by the logits predictions. -->

<!-- #### Uncertainty Estimation via Monte Carlo Dropout
Inspired by [*"Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning"*](https://arxiv.org/abs/1506.02142) *transformer-rankers* provides a function that does prediction with dropout at test time to get stochastic predictions of relevance. Such function provides a mean relevance probability and its respective uncertainty, i.e. variance, as opposed to the standard point estimates of relevance produced by deterministic rankers.

```python
average_logits, uncertainties = trainer.test_with_dropout(num_foward_prediction_passes=10)
``` -->

## Experimental Results Examples
Validation set results, R<sub>10</sub>@1 values when using BM25 negative sampler (1 negative for train).

### Conversation response ranking

|             | MANtIS | MSDialog | Ubuntu DSTC8-task1 |
|-------------|--------|----------|-----------|
| BERT-ranker | 0.683  | 0.671    | 0.859     |
| T5-ranker |  0.616  |  0.650  |  0.826 |

<!-- ### Similar question ranking

|             | Quora | LinkSO |
|-------------|--------|----------|
| BERT-ranker |   |     |
| T5-ranker |    |    | -->

<!-- 
Passage Retrieval
|             | ANTIQUE | MSMarco |
|-------------|--------|----------|
| BERT-ranker | -  |  -  |
| T5-ranker |  - |  - | -->
