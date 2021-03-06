Quick-start
======================================================

Setup
***********
1. Clone the repository:
::
    git clone https://github.com/Guzpenha/transformer_rankers.git
    cd transformer_rankers

2. Install the library in a virtual env:
::
   python3 -m venv env
   source env/bin/activate
   pip install -r requirements.txt

Example
***********

Fine tune pointwise BERT for Community Question Answering. Run on |colab|.

.. code-block:: python
  :linenos:
   
   from transformers import BertTokenizerFast, BertForSequenceClassification

   from transformer_rankers.trainers import transformer_trainer
   from transformer_rankers.datasets import dataset, preprocess_crr
   from transformer_rankers.negative_samplers import negative_sampling 
   from transformer_rankers.eval import results_analyses_tools
   from transformer_rankers.datasets import downloader

   import pandas as pd
   import logging 

   logging.basicConfig(level=logging.INFO,  format="%(asctime)s [%(levelname)s] %(message)s",
                     handlers=[logging.StreamHandler()])
   
   task = 'qqp'
   data_folder = "./data/"
   logging.info("Starting downloader for task {}".format(task))

   dataDownloader = downloader.DataDownloader(task, data_folder)
   dataDownloader.download_and_preprocess()   

   train = pd.read_csv("./data/{}/train.tsv".format(task), sep="\t")
   valid = pd.read_csv(data_folder+task+"/valid.tsv", sep="\t")

   # Random negative samplers
   ns_train = negative_sampling.RandomNegativeSampler(list(train["question1"].values), 1)
   ns_val = negative_sampling.RandomNegativeSampler(list(valid["question1"].values) + \
      list(train["question1"].values), 1)

   tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

   #Create the loaders for the datasets, with the respective negative samplers        
   dataloader = dataset.QueryDocumentDataLoader(train_df=train, val_df=valid, test_df=valid,
                                 tokenizer=tokenizer, negative_sampler_train=ns_train, 
                                 negative_sampler_val=ns_val, task_type='classification', 
                                 train_batch_size=32, val_batch_size=32, max_seq_len=512, 
                                 sample_data=-1, cache_path="{}/{}".format(data_folder, task))

   train_loader, val_loader, test_loader = dataloader.get_pytorch_dataloaders()


   model = BertForSequenceClassification.from_pretrained('bert-base-cased')

   #Instantiate trainer that handles fitting.
   trainer = transformer_trainer.TransformerTrainer(model=model,train_loader=train_loader,
                                 val_loader=val_loader, test_loader=test_loader, 
                                 num_ns_eval=9, task_type="classification", tokenizer=tokenizer,
                                 validate_every_epochs=1, num_validation_batches=-1,
                                 num_epochs=1, lr=0.0005, sacred_ex=None, 
                                 validate_every_steps=-1, num_training_instances=1000)

   #Train the model
   logging.info("Fitting monoBERT for {}".format(task))
   trainer.fit()

   #Predict for test (in our example the validation set)
   logging.info("Predicting")
   preds, labels, _ = trainer.test()
   res = results_analyses_tools.\
      evaluate_and_aggregate(preds, labels, ['ndcg_cut_10'])

   for metric, v in res.items():
      logging.info("Test {} : {:4f}".format(metric, v))



.. |colab| raw:: html

   <a href=https://colab.research.google.com/drive/1jKTu8UMpG_eAe8RiPS0De4-FLRd49kRf?usp=sharing" target="_blank">Google Colab</a>
