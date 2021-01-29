from transformer_rankers.models import pointwise_bert
from transformer_rankers.trainers import transformer_trainer
from transformer_rankers.datasets import dataset, preprocess_crr
from transformer_rankers.negative_samplers import negative_sampling 
from transformer_rankers.eval import results_analyses_tools
from transformer_rankers.datasets import downloader

from transformers import BertTokenizerFast
import pandas as pd
import logging 


def main():
   logging.basicConfig(level=logging.INFO,  
                     format="%(asctime)s [%(levelname)s] %(message)s",
                     handlers=[logging.StreamHandler()])   
   task = 'qqp'
   data_folder = "../../data/"
   logging.info("Starting downloader for task {}".format(task))

   dataDownloader = downloader.DataDownloader(task, data_folder)
   dataDownloader.download_and_preprocess()   

   train = pd.read_csv("{}/{}/train.tsv".format(data_folder, task), sep="\t")
   valid = pd.read_csv("{}/{}/valid.tsv".format(data_folder, task), sep="\t")

   # Random negative samplers
   ns_train = negative_sampling.RandomNegativeSampler(list(train["question1"].values), 1)
   ns_val = negative_sampling.RandomNegativeSampler(list(valid["question1"].values) + \
      list(train["question1"].values), 1)

   tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

   #Create the loaders for the datasets, with the respective negative samplers        
   dataloader = dataset.QueryDocumentDataLoader(train_df=train, val_df=valid, test_df=valid,
                                 tokenizer=tokenizer, negative_sampler_train=ns_train, 
                                 negative_sampler_val=ns_val, task_type='classification', 
                                 train_batch_size=6, val_batch_size=6, max_seq_len=100, 
                                 sample_data=-1, cache_path="{}/{}".format(data_folder, task))

   train_loader, val_loader, test_loader = dataloader.get_pytorch_dataloaders()


   model = pointwise_bert.BertForPointwiseLearning.from_pretrained('bert-base-cased')

   #Instantiate trainer that handles fitting.
   trainer = transformer_trainer.TransformerTrainer(model=model,train_loader=train_loader,
                                 val_loader=val_loader, test_loader=test_loader, 
                                 num_ns_eval=9, task_type="classification", tokenizer=tokenizer,
                                 validate_every_epochs=1, num_validation_batches=-1,
                                 num_epochs=1, lr=0.0005, sacred_ex=None, 
                                 validate_every_steps=100)

   #Train the model
   logging.info("Fitting pointwise BERT for {}".format(task))
   trainer.fit()

   #Predict for test (in our example the validation set)
   logging.info("Predicting")
   preds, labels, _ = trainer.test()
   res = results_analyses_tools.\
      evaluate_and_aggregate(preds, labels, ['ndcg_cut_10'])

   for metric, v in res.items():
      logging.info("Test {} : {:4f}".format(metric, v))

if __name__ == "__main__":
    main()
 