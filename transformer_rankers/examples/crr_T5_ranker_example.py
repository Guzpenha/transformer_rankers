from transformer_rankers.trainers.transformer_trainer import TransformerTrainer
from transformer_rankers.datasets.crr_dataset import CRRDataLoader
from transformer_rankers.datasets.preprocess_crr import read_crr_tsv_as_df
from transformer_rankers.negative_samplers.negative_sampling import RandomNegativeSampler, BM25NegativeSamplerPyserini, SentenceBERTNegativeSampler

from transformers import T5Tokenizer, T5ForConditionalGeneration
from sacred.observers import FileStorageObserver
from sacred import Experiment
from IPython import embed

import torch
import pandas as pd
import argparse
import logging

ex = Experiment('T5-ranker experiment')

@ex.main
def run_experiment(args):
    args.run_id = str(ex.current_run._id)

    #Load datasets
    add_turn_separator = (args.task != "ubuntu_dstc8") # Ubuntu data has several utterances from same user in the context.
    train = read_crr_tsv_as_df(args.data_folder+args.task+"/train.tsv", args.sample_data, add_turn_separator)
    valid = read_crr_tsv_as_df(args.data_folder+args.task+"/valid.tsv", args.sample_data, add_turn_separator)

    #Choose the negative candidate sampler
    tokenizer = T5Tokenizer.from_pretrained(args.transformer_model)        
    if args.train_negative_sampler == 'random':
        ns_train = RandomNegativeSampler(list(train["response"].values), args.num_ns_train)
    elif args.train_negative_sampler == 'bm25':
        ns_train = BM25NegativeSamplerPyserini(list(train["response"].values), args.num_ns_train, 
                    args.data_folder+args.task+"/anserini/", args.sample_data, args.anserini_folder)
    elif args.train_negative_sampler == 'sentenceBERT':
        ns_train = SentenceBERTNegativeSampler(list(train["response"].values), args.num_ns_train, 
                    args.data_folder+args.task+"/train_sentenceBERTembeds", args.sample_data)        

    if args.test_negative_sampler == 'random':
        ns_val = RandomNegativeSampler(list(valid["response"].values) + list(train["response"].values), args.num_ns_eval)
    elif args.test_negative_sampler == 'bm25':
        ns_val = BM25NegativeSamplerPyserini(list(valid["response"].values) + list(train["response"].values),
                    args.num_ns_eval, args.data_folder+args.task+"/anserini/", args.sample_data, args.anserini_folder)
    elif args.test_negative_sampler == 'sentenceBERT':
        ns_val = SentenceBERTNegativeSampler(list(valid["response"].values) + list(train["response"].values),
                    args.num_ns_eval, args.data_folder+args.task+"/valid_sentenceBERTembeds", args.sample_data)

    #Create the loaders for the datasets, with the respective negative samplers
    dataloader = CRRDataLoader(args=args, train_df=train,
                                val_df=valid, test_df=valid,
                                tokenizer=tokenizer, negative_sampler_train=ns_train,
                                negative_sampler_val=ns_val, task_type='generation')
    train_loader, val_loader, test_loader = dataloader.get_pytorch_dataloaders()


    #Instantiate transformer model to be used
    model = T5ForConditionalGeneration.from_pretrained(args.transformer_model)
    model.resize_token_embeddings(len(dataloader.tokenizer))

    #Instantiate trainer that handles fitting.
    trainer = TransformerTrainer(args, model, train_loader, val_loader, test_loader, 
                                         args.num_ns_eval, "generation", tokenizer)

    #Train
    model_name = model.__class__.__name__
    logging.info("Fitting {} for {}{}".format(model_name, args.data_folder, args.task))
    trainer.fit()

    #Predict for test
    logging.info("Predicting")
    preds = trainer.test()

    #Saving predictions to a file
    preds_df = pd.DataFrame(preds, columns=["prediction_"+str(i) for i in range(len(preds[0]))])
    preds_df.to_csv(args.output_dir+"/"+args.run_id+"/predictions.csv", index=False)

    #Saving model to a file
    if args.save_model:
        torch.save(model.state_dict(), args.output_dir+"/"+args.run_id+"/model")

    return trainer.best_ndcg

def main():
    parser = argparse.ArgumentParser()

    # Input and output configs
    parser.add_argument("--task", default=None, type=str, required=True,
                        help="the task to run T5 ranker for")
    parser.add_argument("--data_folder", default=None, type=str, required=True,
                        help="the folder containing data")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="the folder to output predictions")
    parser.add_argument("--save_model", default=False, type=str, required=False,
                        help="Save trained model at the end of training.")

    #Training procedure
    parser.add_argument("--seed", default=42, type=str, required=False,
                        help="random seed")
    parser.add_argument("--num_epochs", default=100, type=int, required=False,
                        help="Number of epochs for training.")
    parser.add_argument("--max_gpu", default=-1, type=int, required=False,
                        help="max gpu used")
    parser.add_argument("--validate_epochs", default=1, type=int, required=False,
                        help="Run validation every <validate_epochs> epochs.")
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
    parser.add_argument("--train_negative_sampler", default="random", type=str, required=False,
                        help="Negative candidates sampler for training (['random', 'bm25', 'sentenceBERT']) ")
    parser.add_argument("--test_negative_sampler", default="random", type=str, required=False,
                        help="Negative candidates sampler for training (['random', 'bm25', 'sentenceBERT']) ")
    parser.add_argument("--anserini_folder", default="", type=str, required=False,
                        help="Path containing the anserini bin <anserini_folder>/target/appassembler/bin/IndexCollection")

    #Model hyperparameters
    parser.add_argument("--transformer_model", default="t5-small", type=str, required=False,
                        help="Bert model to use (default = t5-small).")
    parser.add_argument("--max_seq_len", default=512, type=int, required=False,
                        help="Maximum sequence length for the inputs.")
    parser.add_argument("--lr", default=5e-6, type=float, required=False,
                        help="Learning rate.")

    args = parser.parse_args()
    args.sacred_ex = ex

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

    ex.observers.append(FileStorageObserver(args.output_dir))
    ex.add_config({'args': args})
    return ex.run()

if __name__ == "__main__":
    main()