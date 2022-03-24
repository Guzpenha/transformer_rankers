from torch.utils.data import DataLoader
from IPython import embed
from tqdm import tqdm

from sentence_transformers import SentenceTransformer,  \
    SentencesDataset, losses, models, util
from sentence_transformers.readers import InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction, RerankingEvaluator

from transformer_rankers.datasets import preprocess_crr 
from transformer_rankers.negative_samplers import negative_sampling 

from convokit import Corpus, download
from functools import reduce

import logging
import argparse
import pandas as pd
import os
import math
import sys
import wandb
import torch

import warnings
warnings.filterwarnings("ignore")

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

    def get_examples(self, filename, ns_name, anserini_folder, sent_bert_model, loss, output_dir, input_pair=True, eval_data=False,
        denoise_negatives=False, num_ns_for_denoising=100, generative_model = 'facebook/blenderbot-3B', remove_cand_subsets=True,
        last_utterance_only=False, use_external_corpus=False):
        """
        filename specified which data split to use (train.csv, dev.csv, test.csv).
        """
        filepath = os.path.join(self.dataset_folder, filename)
        self.data = pd.read_csv(filepath, sep="\t")


        if denoise_negatives:
            num_ns = num_ns_for_denoising
        else:
            num_ns = 10

        candidates = list(self.data["response"].values)
        if use_external_corpus:
            external_datasets = [
                'movie-corpus',
                'wiki-corpus',
                'subreddit-Ubuntu',
                'subreddit-microsoft',
                'subreddit-apple',
                'subreddit-Database',
                'subreddit-DIY',
                'subreddit-electronics',
                'subreddit-ENGLISH',
                'subreddit-gis',
                'subreddit-Physics',
                'subreddit-scifi',
                'subreddit-statistics',
                'subreddit-travel',
                'subreddit-worldbuilding'
            ]
            for ds_name in external_datasets:
                corpus = Corpus(download(ds_name))
                corpus.print_summary_stats()
                for utt in corpus.iter_utterances():
                    if utt.text != "":
                        candidates.append(utt.text)

        if ns_name == "random" or eval_data:
            self.negative_sampler = negative_sampling.RandomNegativeSampler(candidates, num_ns)
        elif ns_name == "bm25":
            index_folder = "/anserini_train_-1/"
            if use_external_corpus:
                index_folder = index_folder.replace("train", "train_expanded_")
            self.negative_sampler = negative_sampling.BM25NegativeSamplerPyserini(candidates, num_ns,
                self.dataset_folder+index_folder, -1, anserini_folder)
        elif ns_name == "sentence_transformer":
            self.negative_sampler = negative_sampling.SentenceBERTNegativeSampler(candidates, num_ns, 
                self.dataset_folder+"/train_sentenceBERTembeds", -1, sent_bert_model, large_index=use_external_corpus)
        elif ns_name == "generative":
            self.negative_sampler = negative_sampling.GenerativeNegativeSamplerForDialogue(num_ns, generative_model)
            
        if loss == 'MarginMSELoss':
            self.negative_sampler.score_relevant_docs = True
        if loss == "ContrastiveLoss" and not eval_data:
            input_pair = False
        if loss == "OnlineContrastiveLoss" and not eval_data:
            input_pair = False
        examples = []
        scores_df = []

        # Code used to annotate some samples
        # samples_to_annotate = []
        # self.data = self.data.sample(200, random_state=42)
        # self.negative_sampler.score_relevant_docs = True
        count_ns_part_of_context = 0
        for idx, row in enumerate(tqdm(self.data.itertuples(index=False), total=len(self.data))):
            context = row[0]
            if last_utterance_only:
                if 'msdialog' in self.dataset_folder:
                    context = context.split("[TURN_SEP]")[-1].split("[UTTERANCE_SEP]")[0].strip()
                else:
                    context = context.split("[TURN_SEP]")[-1].split("[UTTERANCE_SEP]")[-2].strip()

            relevant_response = row[1]
            if not input_pair:
                examples.append(InputExample(guid=filename+str(idx)+"_pos",
                    texts=[context, relevant_response], label=1.0))
            if ns_name == "bm25" and not eval_data:
                ns_candidates, ns_scores , _ , _ , rel_scores = self.negative_sampler.sample(context, [relevant_response], max_query_len = 512, normalize_scores = False, rel_doc_id = str(idx))
            else:
                ns_candidates, ns_scores , _ , _ , rel_scores = self.negative_sampler.sample(context, [relevant_response])
            rel_score = rel_scores[0]

            if denoise_negatives:
                zipped = zip(ns_candidates[-10:], ns_scores[-10:])
            else: 
                zipped = zip(ns_candidates, ns_scores)

            for ns, score_ns in zipped:
                if remove_cand_subsets and ns.replace("<<<AGENT>>>: ", "") in context:
                    count_ns_part_of_context+=1
                else: 
                    if input_pair:
                        examples.append(InputExample(texts=[context, relevant_response, ns], label=float(rel_score-score_ns)))
                        scores_df.append(rel_score-score_ns)
                        # samples_to_annotate.append([self.dataset_folder.split("/")[-1], ns_name, context, relevant_response, ns, rel_score, score_ns])
                    else:
                        examples.append(InputExample(guid=filename+str(idx)+"_neg", 
                            texts=[context, ns], label=0.0))
        logging.info("{} {} count of ns which are part of the context: {} out of {}.".format(self.dataset_folder.split("/")[-1],
         ns_name, count_ns_part_of_context, len(examples)))
        # print(pd.DataFrame(scores_df).describe())
        # pd.DataFrame(samples_to_annotate, columns=['task', 'ns', 'context', 'rel_response', 'negative_sample', 'rel_score', 'score_negative']).\
        #     to_csv(output_dir+"neg_samples_{}_{}.csv".format(ns_name, self.dataset_folder.split("/")[-1]), index=False)        

        if loss == 'MarginMSELoss':
            pd.DataFrame(scores_df).to_csv(output_dir+"MarginScores_{}_{}.csv".format(ns_name, self.dataset_folder.split("/")[-1]))
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

    parser.add_argument('--denoise_negatives', dest='denoise_negatives', action='store_true')
    parser.add_argument('--no-denoise_negatives', dest='denoise_negatives', action='store_false')
    parser.set_defaults(denoise_negatives=False)
    parser.add_argument("--num_ns_for_denoising", default=100, type=int, required=False,
                        help="Only used for --denoise_negatives. Number of total of samples to retrieve and get the bottom 10.")

    parser.add_argument("--generative_sampling_model", default="all-MiniLM-L6-v2", type=str, required=False,
                        help="model to use for generating negative samples on the go.")

    parser.add_argument('--remove_cand_subsets', dest='remove_cand_subsets', action='store_true')
    parser.add_argument('--dont_remove_cand_subsets', dest='remove_cand_subsets', action='store_false')
    parser.set_defaults(remove_cand_subsets=True)

    #which part of the context we use to sample negatives.
    parser.add_argument('--last_utterance_only', dest='last_utterance_only', action='store_true')
    parser.add_argument('--all_utterances', dest='last_utterance_only', action='store_false')
    parser.set_defaults(last_utterance_only=False)

    # External corpus to augment negative sampling
    parser.add_argument('--external_corpus', dest='use_external_corpus', action='store_true')
    parser.add_argument('--dont_use_external_corpus', dest='use_external_corpus', action='store_false')
    parser.set_defaults(use_external_corpus=False)

    # #Training procedure
    parser.add_argument("--num_epochs", default=3, type=int, required=False,
                        help="Number of epochs for training.")
    parser.add_argument("--train_batch_size", default=8, type=int, required=False,
                        help="Training batch size.")
    # #Model hyperparameters
    parser.add_argument("--transformer_model", default="bert-base-cased", type=str, required=False,
                        help="Bert model to use (default = bert-base-cased).")
    parser.add_argument("--loss", default='MultipleNegativesRankingLoss', type=str, required=False,
                        help="Loss function to use ['MultipleNegativesRankingLoss', 'TripletLoss', 'MarginMSELoss']")

    ## Wandb project name 
    parser.add_argument("--wandb_project", default='train_sentence_transformer', type=str, required=False,
                        help="name of the wandb project")
    parser.add_argument("--seed", default=42, type=int, required=False,
                        help="Random seed.")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    max_seq_length = 300
    if args.transformer_model == 'all-mpnet-base-v2' or args.transformer_model == 'msmarco-bert-base-dot-v5':
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


    eval_only = False
    if eval_only:
        logging.info("Skipping training (eval_only=True)")
    
    else:
        logging.info("Creating train CRR dataset for {} using {}.".format(args.task, args.negative_sampler))
        crr_reader = CRRBenchmarkDataReader('{}/{}'.format(args.data_folder, args.task))
        train_data = crr_reader.get_examples("train.tsv", args.negative_sampler,
                                    args.anserini_folder, args.sentence_bert_ns_model, args.loss, args.output_dir,
                                    True, False,
                                    args.denoise_negatives, args.num_ns_for_denoising,
                                    args.generative_sampling_model,
                                    args.remove_cand_subsets,
                                    args.last_utterance_only,
                                    args.use_external_corpus)
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.train_batch_size)
    
    if args.loss == 'MultipleNegativesRankingLoss':
        train_loss = losses.MultipleNegativesRankingLoss(model=model, similarity_fct=util.dot_score)
    elif args.loss == 'MarginMSELoss':
        train_loss = losses.MarginMSELoss(model=model)
    elif args.loss == 'TripletLoss':
        train_loss = losses.TripletLoss(model=model)
    elif args.loss == 'ContrastiveLoss':
        train_loss = losses.ContrastiveLoss(model=model)
    elif args.loss == 'OnlineContrastiveLoss':
        train_loss = losses.OnlineContrastiveLoss(model=model)


    ns_description = args.negative_sampler
    if args.negative_sampler == 'sentence_transformer':
        ns_description+="_{}".format(args.sentence_bert_ns_model)

    if args.negative_sampler == 'generative':
        ns_description+="_{}".format(args.generative_sampling_model)

    wandb.init(project=args.wandb_project)
    wandb.config.update(args)

    if not eval_only: # this is the eval data for the training, not the actual evaluation
        logging.info("Getting eval data")
        examples_dev = crr_reader.get_examples('valid.tsv', 
            args.negative_sampler, args.anserini_folder, args.sentence_bert_ns_model, args.loss, args.output_dir, eval_data=True)
        examples_dev = examples_dev[0:(11*500)]
        eval_samples = []
        docs = []
        for i, example in enumerate(examples_dev):
            if (i+1)%11==0:
                eval_samples.append({'query': example.texts[0], 
                                    'positive': [example.texts[1]],
                                    'negative': docs
                })
                docs=[]
            else:
                docs.append(example.texts[2])
        evaluator = RerankingEvaluator(eval_samples, write_csv=True, similarity_fct=util.dot_score)
        warmup_steps = math.ceil(len(train_data)*args.num_epochs/args.train_batch_size*0.1) #10% of train data for warm-up
        logging.info("Warmup-steps: {}".format(warmup_steps))

        logging.info("Fitting sentenceBERT for {}".format(args.task))

        model.fit(train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=args.num_epochs,
            evaluation_steps=100,          
            steps_per_epoch=10000,        
            warmup_steps=warmup_steps,
            output_path=args.output_dir+"{}_{}_ns_{}_loss_{}".format(args.transformer_model, args.task, ns_description, args.loss))

    logging.info("Evaluating for full retrieval of responses to dialogue.")

    train = pd.read_csv(args.data_folder+args.task+"/train.tsv", sep="\t")
    test = pd.read_csv(args.data_folder+args.task+"/test.tsv", sep="\t")

    ns_test_sentenceBERT = negative_sampling.SentenceBERTNegativeSampler(list(train["response"].values)+list(test["response"].values), 10, 
                   args.data_folder+args.task+"/test_sentenceBERTembeds", -1, 
                   args.output_dir+"{}_{}_ns_{}_loss_{}".format(args.transformer_model, args.task, ns_description, args.loss),
                   use_cache_for_embeddings=False)
    
    ns_info = [
        (ns_test_sentenceBERT, 
        ["cand_sentenceBERT_{}".format(i) for i in range(10)] + ["sentenceBERT_retrieved_relevant", "sentenceBERT_rank"], 
        'sentenceBERT')
    ]
    examples = []
    examples_cols = ["context", "relevant_response"] + \
        reduce(lambda x,y:x+y, [t[1] for t in ns_info])
    logging.info("Retrieving candidates using different negative sampling strategies for {}.".format(args.task))
    recall_df = []
    for idx, row in enumerate(tqdm(test.itertuples(index=False), total=len(test))):
        context = row[0]
        relevant_response = row[1]
        instance = [context, relevant_response]

        for ns, _ , ns_name in ns_info:
            ns_candidates, scores, had_relevant, rank_relevant, _ = ns.sample(context, [relevant_response])
            for ns in ns_candidates:
                instance.append(ns)
            instance.append(had_relevant)
            instance.append(rank_relevant)
            if had_relevant:
                r10 = 1
            else:
                r10 = 0
            if rank_relevant == 0:
                r1 = 1
            else:
                r1 =0
            recall_df.append([r10, r1])
        examples.append(instance)

    recall_df  = pd.DataFrame(recall_df, columns = ["R@10", "R@1"])
    examples_df = pd.DataFrame(examples, columns=examples_cols)
    logging.info("R@10: {}".format(examples_df[[c for c in examples_df.columns if 'retrieved_relevant' in c]].sum()/examples_df.shape[0]))
    wandb.log({'R@10': (examples_df[[c for c in examples_df.columns if 'retrieved_relevant' in c]].sum()/examples_df.shape[0]).values[0]})
    rank_col = [c for c in examples_df.columns if 'rank' in c][0]
    logging.info("R@1: {}".format(examples_df[examples_df[rank_col]==0].shape[0]/examples_df.shape[0]))
    wandb.log({'R@1': examples_df[examples_df[rank_col]==0].shape[0]/examples_df.shape[0]})
    recall_df.to_csv(args.output_dir+"/recall_df_{}_{}_ns_{}_loss_{}.csv".format(args.transformer_model.replace("/", "-"), args.task, ns_description.replace("/", "-"), args.loss), index=False, sep="\t")
    # examples_df.to_csv(args.output_dir+"/negative_samples_{}_sample_{}.csv".format(args.task, args.sample_data), index=False, sep="\t")

if __name__ == "__main__":
    main()