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
import json
import os 
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

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    tokenizer = BertTokenizer.from_pretrained(args.transformer_model)
    #Load datasets
    add_turn_separator = (args.task != "ubuntu_dstc8") # Ubuntu data has several utterances from same user in the context.
    train = preprocess_crr.read_crr_tsv_as_df(args.data_folder+args.task+"/train.tsv", args.sample_data, add_turn_separator)
    valid = preprocess_crr.read_crr_tsv_as_df(args.data_folder+args.task+"/valid.tsv", args.sample_data, add_turn_separator)
    special_tokens_dict = {'additional_special_tokens': ['[UTTERANCE_SEP]', '[TURN_SEP]'] }
    tokenizer.add_special_tokens(special_tokens_dict)

    #Choose the negative candidate sampler
    document_col = train.columns[1]    
    ns_train = negative_sampling.BM25NegativeSamplerPyserini(list(train[document_col].values), args.num_ns_train, 
                args.data_folder+args.task+"/anserini_train/", args.sample_data, args.anserini_folder)

    ns_val_random = negative_sampling.RandomNegativeSampler(list(valid[document_col].values)
                    + list(train[document_col].values), args.num_ns_eval)
    ns_val_bm25 = negative_sampling.BM25NegativeSamplerPyserini(list(valid[document_col].values) + list(train[document_col].values),
                    args.num_ns_eval, args.data_folder+args.task+"/anserini_valid/", args.sample_data, args.anserini_folder) 
    ns_val_bert_sentence = negative_sampling.SentenceBERTNegativeSampler(list(valid[document_col].values) + list(train[document_col].values),
                args.num_ns_eval, args.data_folder+args.task+"/valid_sentenceBERTembeds", args.sample_data, args.bert_sentence_model)

    #Create the loaders for the datasets, with the respective negative samplers
    cross_ns_val = {}
    cross_ns_train = {}
    for (ns_name, ns_val) in [ ("random", ns_val_random),
                                ("bm25", ns_val_bm25),
                                ("sentenceBERT", ns_val_bert_sentence)]:
        dataloader = dataset.QueryDocumentDataLoader(train, valid, valid,
                                    tokenizer, ns_train, ns_val,
                                    'classification', args.train_batch_size, 
                                    args.val_batch_size, args.max_seq_len, 
                                    args.sample_data, args.data_folder + args.task)
        train_loader, val_loader, test_loader = dataloader.get_pytorch_dataloaders()
        cross_ns_val[ns_name] = val_loader
        cross_ns_train[ns_name] = train_loader

    #Instantiate transformer model to be used
    model = BertForSequenceClassification.from_pretrained(args.transformer_model)
    model.resize_token_embeddings(len(dataloader.tokenizer))

    #Instantiate trainer that handles fitting.
    trainer = transformer_trainer.TransformerTrainer(model, cross_ns_train["bm25"], 
                                 cross_ns_val["bm25"], cross_ns_val["bm25"], 
                                 args.num_ns_eval, "classification", tokenizer,
                                 args.validate_every_epochs, args.num_validation_batches,
                                 args.num_epochs, args.lr, args.sacred_ex)

    #Train
    model_name = model.__class__.__name__
    logging.info("Fitting {} for {}{}".format(model_name, args.data_folder, args.task))
    trainer.fit()

    #Cross-NS predictions
    for ns_index, ns_name in enumerate(["random", "bm25", "sentenceBERT"]):
        logging.info("Predicting for NS {}".format(ns_name))
        os.makedirs(args.output_dir+"/"+str(int(args.run_id)+ns_index), exist_ok=True)
        with open(args.output_dir+"/"+str(int(args.run_id)+ns_index)+"/config.json", "w") as f:            
            config_w = {'args': vars(args)}
            config_w['args']['test_dataset'] = args.task
            config_w['args']['train_negative_sampler'] = 'bm25'
            config_w['args']['test_negative_sampler'] = ns_name
            if 'sacred_ex' in config_w['args']:
                del config_w['args']['sacred_ex']
            json.dump(config_w, f, indent=4)
        # preds, labels, softmax_logits = trainer.test()
        trainer.num_validation_batches =-1 # no sample
        preds, labels, softmax_logits = trainer.predict(cross_ns_val[ns_name])

        #Saving predictions and labels to a file
        max_preds_column = max([len(l) for l in preds])
        preds_df = pd.DataFrame(preds, columns=["prediction_"+str(i) for i in range(max_preds_column)])
        preds_df.to_csv(args.output_dir+"/"+str(int(args.run_id)+ns_index)+"/predictions.csv", index=False)

        softmax_df = pd.DataFrame(softmax_logits, columns=["prediction_"+str(i) for i in range(max_preds_column)])
        softmax_df.to_csv(args.output_dir+"/"+str(int(args.run_id)+ns_index)+"/predictions_softmax.csv", index=False)

        labels_df = pd.DataFrame(labels, columns=["label_"+str(i) for i in range(max_preds_column)])
        labels_df.to_csv(args.output_dir+"/"+str(int(args.run_id)+ns_index)+"/labels.csv", index=False)

        #Saving model to a file
        if args.save_model:
            torch.save(model.state_dict(), args.output_dir+"/"+str(int(args.run_id)+ns_index)+"/model")

        #In case we want to get uncertainty estimations at prediction time
        if args.predict_with_uncertainty_estimation:  
            logging.info("Predicting with dropout.")
            trainer.num_validation_batches =-1 # no sample
            preds, labels, softmax_logits, foward_passes_preds, uncertainties = \
                trainer.predict_with_uncertainty(cross_ns_val[ns_name], args.num_foward_prediction_passes)
            
            max_preds_column = max([len(l) for l in preds])
            preds_df = pd.DataFrame(preds, columns=["prediction_"+str(i) for i in range(max_preds_column)])
            preds_df.to_csv(args.output_dir+"/"+str(int(args.run_id)+ns_index)+"/predictions_with_dropout.csv", index=False)

            softmax_df = pd.DataFrame(softmax_logits, columns=["prediction_"+str(i) for i in range(max_preds_column)])
            softmax_df.to_csv(args.output_dir+"/"+str(int(args.run_id)+ns_index)+"/predictions_with_dropout_softmax.csv", index=False)

            for i, f_pass_preds in enumerate(foward_passes_preds):
                preds_df = pd.DataFrame(f_pass_preds, columns=["prediction_"+str(i) for i in range(max_preds_column)])
                preds_df.to_csv(args.output_dir+"/"+str(int(args.run_id)+ns_index)+"/predictions_with_dropout_f_pass_{}.csv".format(i), index=False)

            labels_df = pd.DataFrame(labels, columns=["label_"+str(i) for i in range(max_preds_column)])
            labels_df.to_csv(args.output_dir+"/"+str(int(args.run_id)+ns_index)+"/labels.csv", index=False)
            
            uncertainties_df = pd.DataFrame(uncertainties, columns=["uncertainty_"+str(i) for i in range(max_preds_column)])
            uncertainties_df.to_csv(args.output_dir+"/"+str(int(args.run_id)+ns_index)+"/uncertainties.csv", index=False)

    #Cross-dataset predictions
    cross_datasets = set(["msdialog", "ubuntu_dstc8", "mantis"]) - set([args.task])
    cross_datasets = sorted(list(cross_datasets))
    cross_data_val_dataloader = {}
    for cross_task in cross_datasets:
        train_cross = preprocess_crr.read_crr_tsv_as_df(args.data_folder+cross_task+"/train.tsv", args.sample_data, add_turn_separator)
        valid_cross = preprocess_crr.read_crr_tsv_as_df(args.data_folder+cross_task+"/valid.tsv", args.sample_data, add_turn_separator)
        ns_train_cross = negative_sampling.BM25NegativeSamplerPyserini(list(train_cross[document_col].values), args.num_ns_train, 
                args.data_folder+cross_task+"/anserini_train/", args.sample_data, args.anserini_folder)
        ns_val_bm25_cross = negative_sampling.BM25NegativeSamplerPyserini(list(valid_cross[document_col].values) + list(train_cross[document_col].values),
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
        os.makedirs(args.output_dir+"/"+str(int(args.run_id)+ns_index+task_index+1), exist_ok=True)
        with open(args.output_dir+"/"+str(int(args.run_id)+ns_index+task_index+1)+"/config.json", "w") as f:            
            config_w = {'args': vars(args)}
            config_w['args']['test_dataset'] = cross_task
            config_w['args']['train_negative_sampler'] = 'bm25'
            config_w['args']['test_negative_sampler'] = 'bm25'
            if 'sacred_ex' in config_w['args']:
                del config_w['args']['sacred_ex']
            json.dump(config_w, f, indent=4)
        # preds, labels, softmax_logits = trainer.test()
        trainer.num_validation_batches =-1 # no sample
        preds, labels, softmax_logits = trainer.predict(cross_data_val_dataloader[cross_task])

        #Saving predictions and labels to a file
        max_preds_column = max([len(l) for l in preds])
        preds_df = pd.DataFrame(preds, columns=["prediction_"+str(i) for i in range(max_preds_column)])
        preds_df.to_csv(args.output_dir+"/"+str(int(args.run_id)+ns_index+task_index+1)+"/predictions.csv", index=False)

        softmax_df = pd.DataFrame(softmax_logits, columns=["prediction_"+str(i) for i in range(max_preds_column)])
        softmax_df.to_csv(args.output_dir+"/"+str(int(args.run_id)+ns_index+task_index+1)+"/predictions_softmax.csv", index=False)

        labels_df = pd.DataFrame(labels, columns=["label_"+str(i) for i in range(max_preds_column)])
        labels_df.to_csv(args.output_dir+"/"+str(int(args.run_id)+ns_index+task_index+1)+"/labels.csv", index=False)

        #Saving model to a file
        if args.save_model:
            torch.save(model.state_dict(), args.output_dir+"/"+str(int(args.run_id)+ns_index+task_index+1)+"/model")

        #In case we want to get uncertainty estimations at prediction time
        if args.predict_with_uncertainty_estimation:  
            logging.info("Predicting with dropout.")
            preds, labels, softmax_logits, foward_passes_preds, uncertainties = \
                trainer.predict_with_uncertainty(cross_data_val_dataloader[cross_task], args.num_foward_prediction_passes)
            
            max_preds_column = max([len(l) for l in preds])
            preds_df = pd.DataFrame(preds, columns=["prediction_"+str(i) for i in range(max_preds_column)])
            preds_df.to_csv(args.output_dir+"/"+str(int(args.run_id)+ns_index+task_index+1)+"/predictions_with_dropout.csv", index=False)

            softmax_df = pd.DataFrame(softmax_logits, columns=["prediction_"+str(i) for i in range(max_preds_column)])
            softmax_df.to_csv(args.output_dir+"/"+str(int(args.run_id)+ns_index+task_index+1)+"/predictions_with_dropout_softmax.csv", index=False)

            for i, f_pass_preds in enumerate(foward_passes_preds):
                preds_df = pd.DataFrame(f_pass_preds, columns=["prediction_"+str(i) for i in range(max_preds_column)])
                preds_df.to_csv(args.output_dir+"/"+str(int(args.run_id)+ns_index+task_index+1)+"/predictions_with_dropout_f_pass_{}.csv".format(i), index=False)

            labels_df = pd.DataFrame(labels, columns=["label_"+str(i) for i in range(max_preds_column)])
            labels_df.to_csv(args.output_dir+"/"+str(int(args.run_id)+ns_index+task_index+1)+"/labels.csv", index=False)
            
            uncertainties_df = pd.DataFrame(uncertainties, columns=["uncertainty_"+str(i) for i in range(max_preds_column)])
            uncertainties_df.to_csv(args.output_dir+"/"+str(int(args.run_id)+ns_index+task_index+1)+"/uncertainties.csv", index=False)
    return 0.0

def main():
    parser = argparse.ArgumentParser()

    # Input and output configs
    parser.add_argument("--task", default=None, type=str, required=True,
                        help="the task to run bert ranker for")
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
    parser.add_argument("--num_validation_batches", default=-1, type=int, required=False,
                        help="Run validation for a sample of <num_validation_batches>. To run on all instances use -1.")
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