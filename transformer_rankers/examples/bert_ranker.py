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
import wandb

wandb.init(project="transformer-ranker-tests")
ex = Experiment('pointwise-BERT-ranker experiment')

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
    # Conversation Response Ranking datasets needs special tokens
    if args.task in ["mantis", "msdialog", "ubuntu_dstc8"]: 
        special_tokens_dict = {'additional_special_tokens': ['[UTTERANCE_SEP]', '[TURN_SEP]'] }
        tokenizer.add_special_tokens(special_tokens_dict)        

    #Load datasets
    train = pd.read_csv(args.data_folder+args.task+"/train.tsv", sep="\t", 
                        nrows=args.sample_data if args.sample_data != -1 else None)
    valid = pd.read_csv(args.data_folder+args.task+"/valid.tsv", sep="\t",
                        nrows=args.sample_data if args.sample_data != -1 else None)

    #Choose the negative candidate sampler
    document_col = train.columns[1]
    if args.train_negative_sampler == 'random':
        ns_train = negative_sampling.RandomNegativeSampler(list(train[document_col].values), args.num_ns_train)
    elif args.train_negative_sampler == 'bm25':
        ns_train = negative_sampling.BM25NegativeSamplerPyserini(list(train[document_col].values), args.num_ns_train, 
                    args.data_folder+args.task+"/anserini_train/", args.sample_data, args.anserini_folder)
    elif args.train_negative_sampler == 'sentenceBERT':
        ns_train = negative_sampling.SentenceBERTNegativeSampler(list(train[document_col].values), args.num_ns_train, 
                    args.data_folder+args.task+"/train_sentenceBERTembeds", args.sample_data, args.bert_sentence_model)        

    if args.test_negative_sampler == 'random':
        ns_val = negative_sampling.RandomNegativeSampler(list(valid[document_col].values) + list(train[document_col].values), args.num_ns_eval)
    elif args.test_negative_sampler == 'bm25':
        ns_val = negative_sampling.BM25NegativeSamplerPyserini(list(valid[document_col].values) + list(train[document_col].values),
                    args.num_ns_eval, args.data_folder+args.task+"/anserini_valid/", args.sample_data, args.anserini_folder)
    elif args.test_negative_sampler == 'sentenceBERT':
        ns_val = negative_sampling.SentenceBERTNegativeSampler(list(valid[document_col].values) + list(train[document_col].values),
                    args.num_ns_eval, args.data_folder+args.task+"/valid_sentenceBERTembeds", args.sample_data, args.bert_sentence_model)

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
                                 args.validate_every_epochs, args.num_validation_batches,
                                 args.num_epochs, args.lr, args.sacred_ex, args.validate_every_steps, 
                                 validation_metric='R_10@1')

    #Train
    model_name = model.__class__.__name__
    logging.info("Fitting {} for {}{}".format(model_name, args.data_folder, args.task))
    trainer.fit()

    #Predict for test
    logging.info("Predicting for the validation set.")
    preds, labels, softmax_logits = trainer.test()
    res = results_analyses_tools.evaluate_and_aggregate(preds, labels, ['R_10@1'])
    for metric, v in res.items():
        logging.info("Test {} : {:3f}".format(metric, v))
        wandb.log({'step': 0, "dev_"+metric : v})

    #Saving predictions and labels to a file
    max_preds_column = max([len(l) for l in preds])
    preds_df = pd.DataFrame(preds, columns=["prediction_"+str(i) for i in range(max_preds_column)])
    preds_df.to_csv(args.output_dir+"/"+args.run_id+"/predictions.csv", index=False)

    softmax_df = pd.DataFrame(softmax_logits, columns=["prediction_"+str(i) for i in range(max_preds_column)])
    softmax_df.to_csv(args.output_dir+"/"+args.run_id+"/predictions_softmax.csv", index=False)

    labels_df = pd.DataFrame(labels, columns=["label_"+str(i) for i in range(max_preds_column)])
    labels_df.to_csv(args.output_dir+"/"+args.run_id+"/labels.csv", index=False)

    #Saving model to a file
    if args.save_model:
        torch.save(model.state_dict(), args.output_dir+"/"+args.run_id+"/model")

    #In case we want to get uncertainty estimations at prediction time
    if args.predict_with_uncertainty_estimation:  
        logging.info("Predicting with MC dropout for the validation set.")
        preds, labels, softmax_logits, foward_passes_preds, uncertainties = trainer.test_with_dropout(args.num_foward_prediction_passes)
        res = results_analyses_tools.evaluate_and_aggregate(preds, labels, ['R_10@1'])
        for metric, v in res.items():
            logging.info("Test (w. dropout and {} foward passes) {} : {:3f}".format(args.num_foward_prediction_passes, metric, v))
        
        max_preds_column = max([len(l) for l in preds])
        preds_df = pd.DataFrame(preds, columns=["prediction_"+str(i) for i in range(max_preds_column)])
        preds_df.to_csv(args.output_dir+"/"+args.run_id+"/predictions_with_dropout.csv", index=False)

        softmax_df = pd.DataFrame(softmax_logits, columns=["prediction_"+str(i) for i in range(max_preds_column)])
        softmax_df.to_csv(args.output_dir+"/"+args.run_id+"/predictions_with_dropout_softmax.csv", index=False)

        for i, f_pass_preds in enumerate(foward_passes_preds):
            preds_df = pd.DataFrame(f_pass_preds, columns=["prediction_"+str(i) for i in range(max_preds_column)])
            preds_df.to_csv(args.output_dir+"/"+args.run_id+"/predictions_with_dropout_f_pass_{}.csv".format(i), index=False)

        labels_df = pd.DataFrame(labels, columns=["label_"+str(i) for i in range(max_preds_column)])
        labels_df.to_csv(args.output_dir+"/"+args.run_id+"/labels.csv", index=False)
        
        uncertainties_df = pd.DataFrame(uncertainties, columns=["uncertainty_"+str(i) for i in range(max_preds_column)])
        uncertainties_df.to_csv(args.output_dir+"/"+args.run_id+"/uncertainties.csv", index=False)

    return trainer.best_eval_metric

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
    parser.add_argument("--validate_every_epochs", default=-1, type=int, required=False,
                        help="Run validation every <validate_every_epochs> epochs.")
    parser.add_argument("--validate_every_steps", default=1, type=int, required=False,
                        help="Run validation every <validate_every_steps> steps.")
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
    parser.add_argument("--train_negative_sampler", default="random", type=str, required=False,
                        help="Negative candidates sampler for training (['random', 'bm25', 'sentenceBERT']) ")
    parser.add_argument("--test_negative_sampler", default="random", type=str, required=False,
                        help="Negative candidates sampler for training (['random', 'bm25', 'sentenceBERT']) ")
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
    args.model = "pointwise-BERT-ranker"

    ex.observers.append(FileStorageObserver(args.output_dir))
    ex.add_config({'args': args})
    wandb.config.update(args)
    return ex.run()

if __name__ == "__main__":
    main()