from transformer_rankers.trainers import transformer_trainer
from transformer_rankers.datasets import dataset, preprocess_crr, preprocess_sqr
from transformer_rankers.negative_samplers import negative_sampling 
from transformer_rankers.eval import results_analyses_tools

from transformers import BertTokenizer, BertForSequenceClassification
from sacred.observers import FileStorageObserver
from sacred import Experiment
from IPython import embed

import torch
import pandas as pd
import argparse
import logging
import sys
import pickle
import os
import json

ex = Experiment('BERT-ranker experiment')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

@ex.main
def run_experiment(args):
    args.run_id = str(ex.current_run._id)
    args.task = 'trec2020pr'
    tokenizer = BertTokenizer.from_pretrained(args.transformer_model)
    if args.sample_data == -1: args.sample_data=None
    train = pd.read_csv(args.data_folder+args.task+"/train.tsv", sep="\t", nrows=args.sample_data)
    valid = pd.read_csv(args.data_folder+args.task+"/valid.tsv", sep="\t", nrows=args.sample_data)

    #Choose the negative candidate sampler
    document_col = train.columns[1]
    ns_train = negative_sampling.BM25NegativeSamplerPyserini(list(train[document_col].values), args.num_ns_train, 
            args.data_folder+args.task+"/anserini_train/", args.sample_data, args.anserini_folder)

    ns_val = negative_sampling.BM25NegativeSamplerPyserini(list(valid[document_col].values) + list(train[document_col].values),
                args.num_ns_eval, args.data_folder+args.task+"/anserini_valid/", args.sample_data, args.anserini_folder)

    #Create the loaders for the datasets, with the respective negative samplers
    dataloader = dataset.QueryDocumentDataLoader(train, valid, valid,
                                tokenizer, ns_train, ns_val,
                                'classification', args.train_batch_size, 
                                args.val_batch_size, args.max_seq_len, 
                                args.sample_data, args.data_folder + args.task)

    train_loader, val_loader, test_loader = dataloader.get_pytorch_dataloaders()


    #Instantiate transformer model to be used
    model = BertForSequenceClassification.from_pretrained(args.transformer_model)
    model.resize_token_embeddings(len(dataloader.tokenizer))

    #Instantiate trainer that handles fitting.
    trainer = transformer_trainer.TransformerTrainer(model, train_loader, val_loader, test_loader, 
                                 args.num_ns_eval, "classification", tokenizer,
                                 args.validate_every_epochs, args.num_validation_instances,
                                 args.num_epochs, args.lr, args.sacred_ex)

    #Train
    model_name = model.__class__.__name__
    logging.info("Fitting {} for {}{}".format(model_name, args.data_folder, args.task))
    trainer.fit()

    with open(args.data_folder+"clariq/query_top_10_documents.pkl", "rb") as f:
        query_top_10_documents = pickle.load(f)
    all_documents = []
    for doc_list in query_top_10_documents.values():
        all_documents+=doc_list

    cross_datasets = ['clariq']
    cross_data_val_dataloader = {}
    for cross_task in cross_datasets:
        train_cross = pd.read_csv(args.data_folder+cross_task+"/train.tsv", sep="\t", nrows=args.sample_data)
        valid_cross = pd.read_csv(args.data_folder+cross_task+"/valid.tsv", sep="\t", nrows=args.sample_data)
        train_cross = train_cross.drop_duplicates("query")
        valid_cross = valid_cross.drop_duplicates("query")

        valid_cross = pd.concat([train_cross, valid_cross])
        

        ns_train_cross = negative_sampling.BM25NegativeSamplerPyserini(all_documents, args.num_ns_train, 
                args.data_folder+cross_task+"/anserini_train/", args.sample_data, args.anserini_folder)                
        ns_val_bm25_cross = negative_sampling.BM25NegativeSamplerPyserini(all_documents,
                        args.num_ns_eval, args.data_folder+cross_task+"/anserini_valid/", args.sample_data, args.anserini_folder) 
        dataloader = dataset.QueryDocumentDataLoader(train_cross, valid_cross, valid_cross,
                            tokenizer, ns_train_cross, ns_val_bm25_cross,
                            'classification', args.train_batch_size, 
                            args.val_batch_size, args.max_seq_len, 
                            args.sample_data, args.data_folder + cross_task)
        _, val_loader, _ = dataloader.get_pytorch_dataloaders()
        cross_data_val_dataloader[cross_task] = val_loader

    for task_index, cross_task in enumerate(cross_datasets):
        logging.info("Predicting for dataset {}".format(cross_task))
        os.makedirs(args.output_dir+"/"+str(int(args.run_id)), exist_ok=True)
        with open(args.output_dir+"/"+str(int(args.run_id))+"/config.json", "w") as f:            
            config_w = {'args': vars(args)}
            config_w['args']['test_dataset'] = cross_task
            config_w['args']['train_negative_sampler'] = 'bm25'
            config_w['args']['test_negative_sampler'] = 'bm25'
            if 'sacred_ex' in config_w['args']:
                del config_w['args']['sacred_ex']
            json.dump(config_w, f, indent=4)        

        trainer.num_validation_instances =-1 # no sample
        preds, labels, softmax_logits = trainer.predict(cross_data_val_dataloader[cross_task])

        #Saving predictions and labels to a file
        max_preds_column = max([len(l) for l in preds])
        preds_df = pd.DataFrame(preds, columns=["prediction_"+str(i) for i in range(max_preds_column)])
        preds_df.to_csv(args.output_dir+"/"+str(int(args.run_id))+"/predictions.csv", index=False)

        softmax_df = pd.DataFrame(softmax_logits, columns=["prediction_"+str(i) for i in range(max_preds_column)])
        softmax_df.to_csv(args.output_dir+"/"+str(int(args.run_id))+"/predictions_softmax.csv", index=False)

        labels_df = pd.DataFrame(labels, columns=["label_"+str(i) for i in range(max_preds_column)])
        labels_df.to_csv(args.output_dir+"/"+str(int(args.run_id))+"/labels.csv", index=False)

        #Saving model to a file
        if args.save_model:
            torch.save(model.state_dict(), args.output_dir+"/"+str(int(args.run_id))+"/model")

        #In case we want to get uncertainty estimations at prediction time
        if args.predict_with_uncertainty_estimation:  
            logging.info("Predicting with dropout.")
            preds, labels, softmax_logits, foward_passes_preds, uncertainties = \
                trainer.predict_with_uncertainty(cross_data_val_dataloader[cross_task], args.num_foward_prediction_passes)
            
            max_preds_column = max([len(l) for l in preds])
            preds_df = pd.DataFrame(preds, columns=["prediction_"+str(i) for i in range(max_preds_column)])
            preds_df.to_csv(args.output_dir+"/"+str(int(args.run_id))+"/predictions_with_dropout.csv", index=False)

            softmax_df = pd.DataFrame(softmax_logits, columns=["prediction_"+str(i) for i in range(max_preds_column)])
            softmax_df.to_csv(args.output_dir+"/"+str(int(args.run_id))+"/predictions_with_dropout_softmax.csv", index=False)

            for i, f_pass_preds in enumerate(foward_passes_preds):
                preds_df = pd.DataFrame(f_pass_preds, columns=["prediction_"+str(i) for i in range(max_preds_column)])
                preds_df.to_csv(args.output_dir+"/"+str(int(args.run_id))+"/predictions_with_dropout_f_pass_{}.csv".format(i), index=False)
            
            uncertainties_df = pd.DataFrame(uncertainties, columns=["uncertainty_"+str(i) for i in range(max_preds_column)])
            uncertainties_df.to_csv(args.output_dir+"/"+str(int(args.run_id))+"/uncertainties.csv", index=False)

    return trainer.best_ndcg

def main():
    parser = argparse.ArgumentParser()

    # Input and output configs
    parser.add_argument("--data_folder", default=None, type=str, required=True,
                        help="the folder containing data")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="the folder to output predictions")
    parser.add_argument("--save_model", default=False, type=bool, required=False,
                        help="Save trained model at the end of training.")

    #Training procedure
    parser.add_argument("--seed", default=42, type=str, required=False,
                        help="random seed")
    parser.add_argument("--num_epochs", default=100, type=int, required=False,
                        help="Number of epochs for training.")
    parser.add_argument("--max_gpu", default=-1, type=int, required=False,
                        help="max gpu used")
    parser.add_argument("--validate_every_epochs", default=1, type=int, required=False,
                        help="Run validation every <validate_every_epochs> epochs.")
    parser.add_argument("--num_validation_instances", default=-1, type=int, required=False,
                        help="Run validation for a sample of <num_validation_instances>. To run on all instances use -1.")
    parser.add_argument("--train_batch_size", default=32, type=int, required=False,
                        help="Training batch size.")
    parser.add_argument("--val_batch_size", default=32, type=int, required=False,
                        help="Validation and test batch size.")
    parser.add_argument("--num_ns_train", default=1, type=int, required=False,
                        help="Number of negatively sampled documents to use during training")
    parser.add_argument("--num_ns_eval", default=9, type=int, required=False,
                        help="Number of negatively sampled documents to use during evaluation")
    parser.add_argument("--sample_data", default=-1, type=int, required=False,
                         help="Amount of data to sample for training and eval. If no sampling required use -1.")
    parser.add_argument("--bert_sentence_model", default="bert-base-nli-stsb-mean-tokens", type=str, required=False,
                        help="Pre-trained sentenceBERT model.")
    parser.add_argument("--anserini_folder", default="", type=str, required=False,
                        help="Path containing the anserini bin <anserini_folder>/target/appassembler/bin/IndexCollection")

    #Model hyperparameters
    parser.add_argument("--transformer_model", default="bert-base-cased", type=str, required=False,
                        help="Bert model to use (default = bert-base-cased).")
    parser.add_argument("--max_seq_len", default=512, type=int, required=False,
                        help="Maximum sequence length for the inputs.")
    parser.add_argument("--lr", default=5e-6, type=float, required=False,
                        help="Learning rate.")

    #Uncertainty estimation hyperparameters
    parser.add_argument("--predict_with_uncertainty_estimation", default=False, action="store_true", required=False,
                        help="Whether to use dropout at test time to get relevance (mean) and uncertainties (variance).")
    parser.add_argument("--num_foward_prediction_passes", default=10, type=int, required=False,
                        help="Number of foward passes with dropout to obtain mean and variance of predictions. "+
                             "Only used if predict_with_uncertainty_estimation == True.")

    args = parser.parse_args()
    args.sacred_ex = ex

    ex.observers.append(FileStorageObserver(args.output_dir))
    ex.add_config({'args': args})
    return ex.run()

if __name__ == "__main__":
    main()