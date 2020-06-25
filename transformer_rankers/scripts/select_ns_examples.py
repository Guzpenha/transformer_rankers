from IPython import embed
import pandas as pd
import argparse
import logging
import random

logging.basicConfig(
level=logging.INFO,
format="%(asctime)s [%(levelname)s] %(message)s",
handlers=[
    logging.StreamHandler()
]
)

parser = argparse.ArgumentParser()

# Input and output configs
parser.add_argument("--task", default=None, type=str, required=True,
                    help="the task to run bert ranker for")
parser.add_argument("--output_dir", default=None, type=str, required=True,
                    help="the folder with the raw negative_samples")

args = parser.parse_args()

random.seed(1)

examples_df = pd.read_csv(args.output_dir+"_all_negative_samples_{}.csv".format(args.task), sep="\t")
examples_df["len_context"] = examples_df.apply(lambda r: len(r["context"]),axis=1)
examples_df["len_bm25_retrieved_relevant"] = examples_df.apply(lambda r: len(str(r["cand_bm25_0"])),axis=1)
examples_df["url_in_rel_response"] = examples_df.apply(lambda r: "http" in str(r["relevant_response"]),axis=1)
logging.info("Task: {}".format(args.task))
for column in ["random_retrieved_relevant", "bm25_retrieved_relevant", "sentenceBERT_retrieved_relevant"]:
    logging.info("column {} : {:.6f} ranked the relevant at the first position (R@1)".format(column, sum(examples_df[column])/examples_df.shape[0]))

samples = examples_df[examples_df["len_context"] < 512][examples_df["len_bm25_retrieved_relevant"] < 512][~examples_df["url_in_rel_response"]].sample(3)\
    [["context", "relevant_response", "cand_random_0", "cand_bm25_0", "cand_sentenceBERT_0", "random_retrieved_relevant", "bm25_retrieved_relevant", "sentenceBERT_retrieved_relevant"]]
samples.to_csv(args.output_dir+"_sample_negative_samples_{}.csv".format(args.task), sep="\t", index=False)