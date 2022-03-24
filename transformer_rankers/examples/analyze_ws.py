from transformer_rankers.trainers import transformer_trainer
from transformer_rankers.datasets import dataset, preprocess_crr, preprocess_sqr
from transformer_rankers.negative_samplers import negative_sampling 
from transformer_rankers.eval import results_analyses_tools
from transformer_rankers.models import pointwise_bert

from transformers import BertTokenizerFast, BertForSequenceClassification
from sacred.observers import FileStorageObserver
from sacred import Experiment
from IPython import embed
from tqdm import tqdm

import torch
import pandas as pd
import argparse
import logging
import sys
import wandb

logging_level = logging.INFO
logging_fmt = "%(asctime)s [%(levelname)s] %(message)s"
try:
    root_logger = logging.getLogger()
    root_logger.setLevel(logging_level)
    root_handler = root_logger.handlers[0]
    root_handler.setFormatter(logging.Formatter(logging_fmt))
except IndexError:
    logging.basicConfig(level=logging_level, format=logging_fmt)

ex = Experiment('pointwise-BERT-ranker experiment')

@ex.main
def run_experiment(args):
    args.run_id = str(ex.current_run._id)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # tokenizer = BertTokenizer.from_pretrained(args.transformer_model)
    tokenizer = BertTokenizerFast.from_pretrained(args.transformer_model)
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

    train = train.groupby(train.columns[0]).agg(list).reset_index()
    labels = []
    sample = 10000
    max_labels = 0
    for idx, row in enumerate(tqdm(train[0:sample].itertuples(index=False), total=sample)):
        query = row[0]
        relevant_documents = row[1]
        query_labels = []
        for relevant_document in relevant_documents:
            query_labels.append(1.0)
        ns_candidates, ns_scores, _, _, _= ns_train.sample(query, relevant_documents)
        for i, ns in enumerate(ns_candidates):
            query_labels.append(ns_scores[i])
        labels.append(query_labels)
        if max_labels < len(query_labels):
            max_labels = len(query_labels)
    df_labels = pd.DataFrame(labels, columns = ["candidate_{}".format(i) for i in range(max_labels)])
    df_labels.to_csv(args.output_dir+"/{}_weak_supervision.csv".format(args.task), index=False)


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
                        help="Number of epochs for training (if num_training_instances != -1 then num_epochs is ignored).")
    parser.add_argument("--num_training_instances", default=-1, type=int, required=False,
                        help="Number of training instances for training (if num_training_instances != -1 then num_epochs is ignored).")
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
                        help="Pre-trained sentenceBERT model (used for NegativeSampling when NS=sentenceBERT).")
    parser.add_argument("--anserini_folder", default="", type=str, required=False,
                        help="Path containing the anserini bin <anserini_folder>/target/appassembler/bin/IndexCollection")

    #Model hyperparameters
    parser.add_argument("--transformer_model", default="bert-base-cased", type=str, required=False,
                        help="Bert model to use (default = bert-base-cased).")
    parser.add_argument("--max_seq_len", default=512, type=int, required=False,
                        help="Maximum sequence length for the inputs.")
    parser.add_argument("--lr", default=5e-6, type=float, required=False,
                        help="Learning rate.")
    parser.add_argument("--loss_function", default="cross-entropy", type=str, required=False,
                        help="Loss function (default is 'cross-entropy').")
    parser.add_argument("--smoothing", default=0.1, type=float, required=False,
                        help="Smoothing hyperparameter used only if loss_function is label-smoothing-cross-entropy.")
    parser.add_argument("--use_ls_cl", default=False, action="store_true", required=False,
                        help="Use curriculum learning for the label smoothing rate.")
    parser.add_argument("--use_ls_ts", default=False, action="store_true", required=False,
                        help="Use two stage (x, 0) for the label smoothing rate.")
    parser.add_argument("--num_instances_TSLA", default=0, type=int, required=False,
                        help="Number of training instances to swap from Label smoothing to one hot.")

    #Uncertainty estimation hyperparameters
    parser.add_argument("--predict_with_uncertainty_estimation", default=False, action="store_true", required=False,
                        help="Whether to use dropout at test time to get relevance (mean) and uncertainties (variance).")
    parser.add_argument("--num_foward_prediction_passes", default=10, type=int, required=False,
                        help="Number of foward passes with dropout to obtain mean and variance of predictions. "+
                             "Only used if predict_with_uncertainty_estimation == True.")

    #Wandb loggging config
    parser.add_argument("--wandb_project", default="wandb-local-run", type=str, required=False,
            help="Wandb project to log.")

    args = parser.parse_args()
    args.sacred_ex = ex
    args.model = "pointwise-BERT-ranker"

    ex.observers.append(FileStorageObserver(args.output_dir))
    ex.add_config({'args': args})

    wandb.init(project=args.wandb_project)
    wandb.config.update(args)   
    return ex.run()

if __name__ == "__main__":
    main()