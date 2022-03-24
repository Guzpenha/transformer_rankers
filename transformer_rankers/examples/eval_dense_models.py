from IPython import embed
from transformer_rankers.negative_samplers import negative_sampling
from transformer_rankers.datasets import preprocess_crr 
from tqdm import tqdm
from functools import reduce


import pandas as pd
import argparse
import logging

import warnings
warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser()

    # Input and output configs
    parser.add_argument("--task", default=None, type=str, required=True,
                        help="the task to run bert ranker for")
    parser.add_argument("--data_folder", default=None, type=str, required=True,
                        help="the folder containing data")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="the folder to output raw negative_samples")
    parser.add_argument("--anserini_folder", default="", type=str, required=True,
                        help="Path containing the anserini bin <anserini_folder>/target/appassembler/bin/IndexCollection")
    parser.add_argument("--sample_data", default=-1, type=int, required=False,
                         help="Amount of data to sample for training and eval. If no sampling required use -1.")
    parser.add_argument("--seed", default=42, type=str, required=False,
                        help="random seed")
    parser.add_argument("--num_ns", default=1, type=int, required=False,
                        help="Number of negatively sampled documents to use during training")
    parser.add_argument("--sentence_bert_model", type=str, required=False, default="all-MiniLM-L6-v2",
                        help="Model to calculate sentence embeddings with for sentenceBERT negative sampling.")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )
    #Load datasets
    train = pd.read_csv(args.data_folder+args.task+"/train.tsv", sep="\t", 
                        nrows=args.sample_data if args.sample_data != -1 else None)
    test = pd.read_csv(args.data_folder+args.task+"/test.tsv", sep="\t",
                        nrows=args.sample_data if args.sample_data != -1 else None)
    # embed()

    # ns_test_random = negative_sampling.RandomNegativeSampler(list(train["response"].values)+list(test["response"].values), args.num_ns)    
    # ns_test_bm25 = negative_sampling.BM25NegativeSamplerPyserini(list(train["response"].values)+list(test["response"].values), args.num_ns,
    #             args.data_folder+args.task+"/anserini_test_{}/".format(args.sample_data), args.sample_data, args.anserini_folder)
    # ns_test_bm25_rm3 = negative_sampling.BM25NegativeSamplerPyserini(list(train["response"].values)+list(test["response"].values), args.num_ns,
    #            args.data_folder+args.task+"/anserini_test_{}/".format(args.sample_data), args.sample_data, args.anserini_folder, set_rm3=True)    
    ns_test_sentenceBERT = negative_sampling.SentenceBERTNegativeSampler(list(train["response"].values)+list(test["response"].values), args.num_ns, 
                   args.data_folder+args.task+"/test_sentenceBERTembeds", args.sample_data, args.sentence_bert_model,
                   use_cache_for_embeddings=False)

    ns_info = [
        # (ns_test_random, ["cand_random_{}".format(i) for i in range(args.num_ns)] + ["random_retrieved_relevant", "random_rank"], 'random'),
        # (ns_test_bm25, ["cand_bm25_{}".format(i) for i in range(args.num_ns)] + ["bm25_retrieved_relevant", "bm25_rank"], 'bm25'),
        # (ns_test_bm25_rm3,["cand_bm25rm3_{}".format(i) for i in range(args.num_ns)] + ["bm25rm3_retrieved_relevant", "bm25rm3_rank"], 'bm25rm3'),        
        (ns_test_sentenceBERT, ["cand_sentenceBERT_{}".format(i) for i in range(args.num_ns)] + ["sentenceBERT_retrieved_relevant", "sentenceBERT_rank"], 'sentenceBERT')
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
    recall_df.to_csv(args.output_dir+"/recall_df_{}_{}.csv".format(args.task, args.sentence_bert_model), index=False, sep="\t")
    examples_df = pd.DataFrame(examples, columns=examples_cols)
    print("R@10: {}".format(examples_df[[c for c in examples_df.columns if 'retrieved_relevant' in c]].sum()/examples_df.shape[0]))
    rank_col = [c for c in examples_df.columns if 'rank' in c][0]
    print("R@1: {}".format(examples_df[examples_df[rank_col]==0].shape[0]/examples_df.shape[0]))
    examples_df.to_csv(args.output_dir+"/negative_samples_{}_sample_{}.csv".format(args.task, args.sample_data), index=False, sep="\t")    

if __name__ == "__main__":
    main()
