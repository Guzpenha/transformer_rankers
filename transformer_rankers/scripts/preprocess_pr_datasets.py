from transformer_rankers.datasets import preprocess_pr

data_path = "../../data/"

# Deep Learning TREC 2020 Passage Ranking
train, valid, test = preprocess_pr.\
    transform_trec2020pr_to_dfs(data_path+"trec2020pr")
train.to_csv(data_path+"trec2020pr/train.tsv", sep="\t", index=False)
valid.to_csv(data_path+"trec2020pr/valid.tsv", sep="\t", index=False)
test.to_csv(data_path+"trec2020pr/test.tsv", sep="\t", index=False)