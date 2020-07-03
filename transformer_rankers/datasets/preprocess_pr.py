from IPython import embed
from tqdm import tqdm

import csv
import gzip
import codecs

import pandas as pd

def transform_trec2020pr_to_dfs(path):
    """
    Transforms TREC 2020 Passage Ranking files (https://microsoft.github.io/TREC-2020-Deep-Learning/)
    to train, valid and test dfs containing only positive query-passage combinations.

    Args:
        path: str with the path for the TREC folder containing: 
            - collection.tar.gz (uncompressed: collection.tsv)
            - queries.tar.gz (uncompressed: queries.train.tsv, queries.dev.tsv)
            - qrels.dev.tsv
            - qrels.train.tsv
        
    Returns: (train, valid, test) pandas DataFrames
    """
    query_df_train = pd.read_csv("{}/queries.train.tsv".format(path), names=['qid','query_string'], sep='\t')
    query_df_train['qid'] = query_df_train['qid'].astype(int)
    queries_str_train = query_df_train.set_index('qid').to_dict()['query_string']

    query_df_dev = pd.read_csv("{}/queries.dev.tsv".format(path), names=['qid','query_string'], sep='\t')
    query_df_dev['qid'] = query_df_dev['qid'].astype(int)
    queries_str_dev = query_df_dev.set_index('qid').to_dict()['query_string']

    collection_str = pd.read_csv("{}/collection.tsv".format(path), sep='\t', names=['docid', 'document_string'])\
        .set_index('docid').to_dict()['document_string']

    qrels_train = pd.read_csv("{}/qrels.train.tsv".format(path), sep="\t", names=["topicid", "_", "docid", "rel"])
    qrels_dev = pd.read_csv("{}/qrels.dev.tsv".format(path), sep="\t", names=["topicid", "_", "docid", "rel"])

    train = []
    for idx, row in tqdm(qrels_train.sort_values("topicid").iterrows()):
        train.append([queries_str_train[row["topicid"]], collection_str[row["docid"]]])
    train_df = pd.DataFrame(train, columns=["query", "passage"])

    dev = []
    for idx, row in tqdm(qrels_dev.sort_values("topicid").iterrows()):
        dev.append([queries_str_dev[row["topicid"]], collection_str[row["docid"]]])
    all_dev_df = pd.DataFrame(dev, columns=["query", "passage"])

    dev_df, test_df = all_dev_df[0:all_dev_df.shape[0]//2], all_dev_df[all_dev_df.shape[0]//2:]

    return train_df, dev_df, test_df