Quick-start
======================================================

Setup
***********
1. Clone the repository:
::
    git clone git@github.com:Guzpenha/transformer_rankers.git    
    cd transformer_rankers

2. [Optional] Create a virtual env (python >= 3.6) and activate it:
::
   python3 -m venv env
   source env/bin/activate

3. Install the library:
::
    pip install -e .
   
4. Install the requirements:
::
   pip install -r requirements.txt

Example (I) - Supported dataset
***********

1. Download and preprocess Similar Question Retrieval data:
::
   cd transformer_rankers/scripts
   ./download_sqr_data.sh

2. Train BERT-ranker for Quora Question Pairs (with only 1000 samples to be fast):
::
   python ../examples/crr_bert_ranker_example.py \
      --task qqp \
      --data_folder ../../data/ \
      --output_dir ../../data/output_data \
      --sample_data 1000

The output will be something like this:
:: 
   [...]
   2020-06-23 11:19:44,522 [INFO] Epoch 1 val nDCG@10: 0.245
   2020-06-23 11:19:44,522 [INFO] Predicting
   2020-06-23 11:19:44,523 [INFO] Starting evaluation on test.
   2020-06-23 11:20:03,678 [INFO] Test ndcg_cut_10: 0.3236

3. The experiment info will be saved at *../data/output_data*, where you can find the following files:
::

   /data/output_data/1/config.json
   /data/output_data/1/cout.txt
   /data/output_data/1/labels.csv
   /data/output_data/1/predictions.csv
   /data/output_data/1/run.json

4. You can easily aggregate the results of different experiment runs using */examples/crr_results_analyses_example.py*:


Example (II) - Custom dataset
***********

In the example below we read a custom .csv with the dataset we want to train the transformer for.

.. code-block:: python
  :linenos:

   from transformer_rankers.trainers import transformer_trainer
   from transformer_rankers.datasets import dataset, preprocess_crr
   from transformer_rankers.negative_samplers import negative_sampling 
   from transformer_rankers.eval import results_analyses_tools

   #Read dataset (the expected format of this tsv is
   #  ["Query", "Relevant_Document"] )
   data_folder = "data"
   task = "custom_dataset"
   train = pd.read_csv(args.data_folder+args.task+"/train.csv")
   valid = pd.read_csv(args.data_folder+args.task+"/valid.csv")

   #Instantiate random negative samplers 
   # (1 for training 9 negative candidates for test)
   ns_train = negative_sampling.\
      RandomNegativeSampler(list(train["response"].values), 1)
   ns_val = negative_sampling.\
      RandomNegativeSampler(list(valid["response"].values) + \
      list(train["response"].values), 9)

   #Create the loaders for the datasets, 
   #with the respective negative samplers        
   tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
   dataloader = dataset.QueryDocumentDataLoader(train_df=train, 
      val_df=valid, test_df=valid,
      tokenizer=tokenizer, negative_sampler_train=ns_train, 
      negative_sampler_val=ns_val, task_type='classification', 
      train_batch_size=32, val_batch_size=32, max_seq_len=512, 
      sample_data=-1, "data/custom_dataset")

   train_loader, val_loader, test_loader = dataloader.\
      get_pytorch_dataloaders()

   #Use BERT to rank responses
   model = BertForSequenceClassification.from_pretrained('bert-base-cased')

   #Instantiate trainer that handles fitting.
   trainer = transformer_trainer.TransformerTrainer(model=model,
      train_loader=train_loader,
      val_loader=val_loader, test_loader=test_loader, 
      num_ns_eval=9, task_type="classification", tokenizer=tokenizer,
      validate_every_epoch=1, num_validation_instances=-1,
      num_epochs=1, lr=0.0005, sacred_ex=None)

   #Train and evaluate
   trainer.fit()
   preds, labels = trainer.test()
   res = results_analyses_tools.\
      evaluate_and_aggregate(preds, labels, ['ndcg_cut_10'])
   for metric, v in res.items():
      logging.info("Test {} : {:4f}".format(metric, v))