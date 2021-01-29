from IPython import embed
from transformer_rankers.negative_samplers import negative_sampling
from transformer_rankers.datasets import preprocess_crr 
from transformers import BertTokenizer
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
    parser.add_argument("--anserini_folder", default="", type=str, required=False,
                        help="Path containing the anserini bin <anserini_folder>/target/appassembler/bin/IndexCollection")
    parser.add_argument("--sample_data", default=-1, type=int, required=False,
                         help="Amount of data to sample for training and eval. If no sampling required use -1.")
    parser.add_argument("--seed", default=42, type=str, required=False,
                        help="random seed")
    parser.add_argument("--num_ns_train", default=1, type=int, required=False,
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
    add_turn_separator = (args.task != "ubuntu_dstc8") # Ubuntu data has several utterances from same user in the context.
    train = preprocess_crr.read_crr_tsv_as_df(args.data_folder+args.task+"/train.tsv", args.sample_data, add_turn_separator)
    valid = preprocess_crr.read_crr_tsv_as_df(args.data_folder+args.task+"/valid.tsv", args.sample_data, add_turn_separator)

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")        
    ns_valid_random = negative_sampling.RandomNegativeSampler(list(train["response"].values)+list(valid["response"].values), args.num_ns_train)    
    ns_valid_bm25 = negative_sampling.BM25NegativeSamplerPyserini(list(train["response"].values)+list(valid["response"].values), args.num_ns_train,
                args.data_folder+args.task+"/anserini_valid/", args.sample_data, args.anserini_folder, set_rm3=True)
    ns_valid_sentenceBERT = negative_sampling.SentenceBERTNegativeSampler(list(train["response"].values)+list(valid["response"].values), args.num_ns_train, 
                args.data_folder+args.task+"/valid_sentenceBERTembeds", args.sample_data, 
                args.data_folder+args.task+"/bert-base-cased_{}".format(args.task)) #pre-trained embedding

    examples = []
    examples_cols = ["context", "relevant_response"] + \
        ["cand_random_{}".format(i) for i in range(args.num_ns_train)] + \
        ["random_retrieved_relevant", "random_rank"]+  \
        ["cand_bm25_{}".format(i) for i in range(args.num_ns_train)] + \
        ["bm25_retrieved_relevant", "bm25_rank"]+  \
        ["cand_sentenceBERT_{}".format(i) for i in range(args.num_ns_train)] + \
        ["sentenceBERT_retrieved_relevant", "sentenceBERT_rank"]
    
    logging.info("Retrieving candidates using random, bm25 and sentenceBERT.")
    for idx, row in enumerate(tqdm(valid.itertuples(index=False), total=len(valid))):
        context = row[0]
        relevant_response = row[1]
        instance = [context, relevant_response]

        for ns_name, ns in [("random", ns_valid_random),
                            ("bm25", ns_valid_bm25),
                            ("sentenceBERT", ns_valid_sentenceBERT)]:
            ns_candidates, scores, had_relevant, rank_relevant = ns.sample(context, relevant_response)
            for ns in ns_candidates:
                instance.append(ns)
            instance.append(had_relevant)
            instance.append(rank_relevant)
        examples.append(instance)

    examples_df = pd.DataFrame(examples, columns=examples_cols)
    examples_df.to_csv(args.output_dir+"/_all_negative_samples_{}.csv".format(args.task), index=False, sep="\t")

if __name__ == "__main__":
    main()