from torch.utils.data import DataLoader
from IPython import embed
from tqdm import tqdm

from sentence_transformers import SentenceTransformer,  \
    SentencesDataset, losses, models, util
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

    def get_examples(self, filename, ns_name, anserini_folder, sent_bert_model, input_pair=True):
        """
        filename specified which data split to use (train.csv, dev.csv, test.csv).
        """
        filepath = os.path.join(self.dataset_folder, filename)
        self.data = pd.read_csv(filepath, sep="\t")
        num_ns = 10
        if ns_name == "random":
            self.negative_sampler = negative_sampling.RandomNegativeSampler(list(self.data["response"].values), num_ns)
        elif ns_name == "bm25":
            self.negative_sampler = negative_sampling.BM25NegativeSamplerPyserini(list(self.data["response"].values), num_ns,
                self.dataset_folder+"/anserini_train_-1/", -1, anserini_folder)
        elif ns_name == "sentence_transformer":
            self.negative_sampler = negative_sampling.SentenceBERTNegativeSampler(list(self.data["response"].values), num_ns, 
                self.dataset_folder+"/train_sentenceBERTembeds", -1, sent_bert_model)

        examples = []
        scores_df = []
        for idx, row in enumerate(tqdm(self.data.itertuples(index=False), total=len(self.data))):
            context = row[0]
            relevant_response = row[1]
            if not input_pair:
                examples.append(InputExample(guid=filename+str(idx)+"_pos",
                    texts=[context, relevant_response], label=1.0))
            ns_candidates, ns_scores , _ , _ = self.negative_sampler.sample(context, relevant_response)
            for ns, score in zip(ns_candidates, ns_scores):
                if input_pair:
                    examples.append(InputExample(texts=[context, relevant_response, ns], label=float(1-score)))
                    scores_df.append(1-score)
                else:
                    examples.append(InputExample(guid=filename+str(idx)+"_neg", 
                        texts=[context, ns], label=0.0))
        # print(pd.DataFrame(scores_df).describe())
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
    parser.add_argument("--negative_sampler", default="random", type=str, required=False,
                        help="negative sampling procedure to use ['random', 'bm25', 'sentence_transformer']")
    parser.add_argument("--anserini_folder", default="", type=str, required=True,
                        help="Path containing the anserini bin <anserini_folder>/target/appassembler/bin/IndexCollection")
    parser.add_argument("--sentence_bert_ns_model", default="all-MiniLM-L6-v2", type=str, required=False,
                        help="model to use for sentenceBERT negative sampling.")
    # #Training procedure
    parser.add_argument("--num_epochs", default=3, type=int, required=False,
                        help="Number of epochs for training.")
    parser.add_argument("--train_batch_size", default=8, type=int, required=False,
                        help="Training batch size.")
    # #Model hyperparameters
    parser.add_argument("--transformer_model", default="bert-base-cased", type=str, required=False,
                        help="Bert model to use (default = bert-base-cased).")
    parser.add_argument("--loss", default='MultipleNegativesRankingLoss', type=str, required=False,
                        help="Loss function to use ['MultipleNegativesRankingLoss' or 'MarginMSELoss']")

    args = parser.parse_args()

    max_seq_length = 300
    if args.transformer_model == 'all-mpnet-base-v2':
        model = SentenceTransformer(args.transformer_model)
        model.max_seq_length = max_seq_length
    else:
        word_embedding_model = models.Transformer(args.transformer_model, max_seq_length=max_seq_length)
        tokens = ['[UTTERANCE_SEP]', '[TURN_SEP]', '[AUG]']
        word_embedding_model.tokenizer.add_tokens(tokens, special_tokens=True)
        word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                pooling_mode_mean_tokens=True,
                                pooling_mode_cls_token=False,
                                pooling_mode_max_tokens=False)
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    logging.info("Creating train CRR dataset for {} using {}.".format(args.task, args.negative_sampler))
    crr_reader = CRRBenchmarkDataReader('{}/{}'.format(args.data_folder, args.task))

    # train_data = SentencesDataset(crr_reader.get_examples("train.tsv", args.negative_sampler,
    #                              args.anserini_folder, args.sentence_bert_ns_model), model)
    train_data = crr_reader.get_examples("train.tsv", args.negative_sampler,
                                 args.anserini_folder, args.sentence_bert_ns_model)
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.train_batch_size)
    
    if args.loss == 'MultipleNegativesRankingLoss':
        train_loss = losses.MultipleNegativesRankingLoss(model=model, similarity_fct=util.dot_score)
    elif args.loss == 'MarginMSELoss':
        train_loss = losses.MarginMSELoss(model=model)

    logging.info("Getting eval data")
    examples_dev = crr_reader.get_examples('valid.tsv', args.negative_sampler, args.anserini_folder, args.sentence_bert_ns_model)
    examples_dev = examples_dev[0:100]
    sentences1 = [v.texts[0] for v in examples_dev]
    sentences2 = [v.texts[1] for v in examples_dev]
    scores = [v.label for v in examples_dev]
    evaluator = EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)

    warmup_steps = math.ceil(len(train_data)*args.num_epochs/args.train_batch_size*0.1) #10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))

    logging.info("Fitting sentenceBERT for {}".format(args.task))
    ns_description = args.negative_sampler
    if args.negative_sampler == 'sentence_transformer':
        ns_description+="_{}".format(args.sentence_bert_ns_model)

    model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=args.num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=args.output_dir+"{}_{}_ns_{}_loss_{}".format(args.transformer_model, args.task, ns_description, args.loss))

if __name__ == "__main__":
    main()