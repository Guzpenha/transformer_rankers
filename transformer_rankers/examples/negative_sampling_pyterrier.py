from IPython import embed
from transformer_rankers.negative_samplers import negative_sampling_pyterrier
from transformer_rankers.datasets import preprocess_crr 
from tqdm import tqdm

import pandas as pd
import argparse
import logging

def main():
    parser = argparse.ArgumentParser()

    # Input and output configs
    parser.add_argument("--task", default=None, type=str, required=True,
                        help="the task to run bert ranker for")
    parser.add_argument("--data_folder", default=None, type=str, required=True,
                        help="the folder containing data")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="the folder to output raw negative_samples")
    parser.add_argument("--sample_data", default=-1, type=int, required=False,
                         help="Amount of data to sample for training and eval. If no sampling required use -1.")
    parser.add_argument("--seed", default=42, type=str, required=False,
                        help="random seed")
    parser.add_argument("--num_ns", default=1, type=int, required=False,
                        help="Number of negatively sampled documents to use during training")
    
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

    ns_test_bm25_pyterrier = negative_sampling_pyterrier.BM25NegativeSamplerPyterrier(list(train["response"].values)+list(test["response"].values), args.num_ns,
                args.data_folder+args.task+"/pyterrier_test/", args.sample_data)

    examples = []
    examples_cols = ["context", "relevant_response"] + \
        ["cand_bm25_pyterrier_{}".format(i) for i in range(args.num_ns)] + \
        ["bm25_pyterrier_retrieved_relevant", "bm25_pyterrier_rank"]

    logging.info("Retrieving candidates using different negative sampling strategies for {}.".format(args.task))
    for idx, row in enumerate(tqdm(test.itertuples(index=False), total=len(test))):
        context = row[0]
        relevant_response = row[1]
        instance = [context, relevant_response]

        for ns_name, ns in [("bm25_pyterrier", ns_test_bm25_pyterrier)]:
            ns_candidates, scores, had_relevant, rank_relevant = ns.sample(context, relevant_response)
            for ns in ns_candidates:
                instance.append(ns)
            instance.append(had_relevant)
            instance.append(rank_relevant)
        examples.append(instance)

    examples_df = pd.DataFrame(examples, columns=examples_cols)
    print("R@10")    
    print(examples_df[["bm25_pyterrier_retrieved_relevant"]].sum()/examples_df.shape[0])
    print("R@1")
    print(examples_df[examples_df["bm25_pyterrier_rank"]==0].shape[0]/examples_df.shape[0])    
    examples_df.to_csv(args.output_dir+"/negative_samples_{}_pyterrier_sample_{}.csv".format(args.task, args.sample_data), index=False, sep="\t")

if __name__ == "__main__":
    main()