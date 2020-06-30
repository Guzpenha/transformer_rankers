from torch.utils.data import DataLoader
from IPython import embed
from tqdm import tqdm

from sentence_transformers import SentenceTransformer,  \
    SentencesDataset, losses, models
from sentence_transformers.readers import InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from transformer_rankers.datasets import preprocess_crr 
from transformer_rankers.negative_samplers import negative_sampling 

import logging
import argparse
import pandas as pd
import os
import math
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)


class CRRDataReader:
    """
    Reads in the CRR dataset.
    """
    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder

    def get_examples(self, filename, max_examples=0):
        """
        filename specified which data split to use (train.csv, dev.csv, test.csv).
        """
        filepath = os.path.join(self.dataset_folder, filename)
        self.data = preprocess_crr.read_crr_tsv_as_df(filepath)
        self.negative_sampler = negative_sampling.RandomNegativeSampler(
            list(self.data["response"].values), 1)
        examples = []
        for idx, row in enumerate(tqdm(self.data.itertuples(index=False), total=len(self.data))):
            context = row[0]
            relevant_response = row[1]
            examples.append(InputExample(guid=filename+str(idx)+"_pos", 
                texts=[context, relevant_response], label=1.0))
            ns_candidates, _, _ = self.negative_sampler.sample(context, relevant_response)
            for ns in ns_candidates:
                examples.append(InputExample(guid=filename+str(idx)+"_neg", 
                    texts=[context, ns], label=0.0))        
        return examples

class CRRBenchmarkDataReader(CRRDataReader):
    """
    Reader especially for CRR (conversation response ranking) datasets.
    """
    def __init__(self, dataset_folder):
        super().__init__(dataset_folder=dataset_folder)

def main():
    parser = argparse.ArgumentParser()

    # Input and output configs
    parser.add_argument("--task", default=None, type=str, required=True,
                        help="the task to run bert ranker for")
    parser.add_argument("--data_folder", default=None, type=str, required=True,
                        help="the folder containing data")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="the folder to output predictions")

    # #Training procedure
    parser.add_argument("--num_epochs", default=5, type=int, required=False,
                        help="Number of epochs for training.")
    parser.add_argument("--train_batch_size", default=8, type=int, required=False,
                        help="Training batch size.")
    # #Model hyperparameters
    parser.add_argument("--transformer_model", default="bert-base-cased", type=str, required=False,
                        help="Bert model to use (default = bert-base-cased).")    


    args = parser.parse_args()

    word_embedding_model = models.Transformer(args.transformer_model)

    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    logging.info("Creating train CRR dataset.")
    crr_reader = CRRBenchmarkDataReader('{}/{}'.format(args.data_folder, args.task))

    train_data = SentencesDataset(crr_reader.get_examples("train.tsv"), model)
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.train_batch_size)
    train_loss = losses.CosineSimilarityLoss(model=model)


    logging.info("Creating dev CRR dataset.")
    dev_data = SentencesDataset(crr_reader.get_examples('valid.tsv'), model)
    dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=args.train_batch_size)
    evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)

    warmup_steps = math.ceil(len(train_data)*args.num_epochs/args.train_batch_size*0.1) #10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))

    logging.info("Fitting sentenceBERT")
    model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=args.num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=args.output_dir+"{}_{}".format(args.transformer_model, args.task))

if __name__ == "__main__":
    main()