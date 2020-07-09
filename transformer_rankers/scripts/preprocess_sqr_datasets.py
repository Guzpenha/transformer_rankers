from transformer_rankers.datasets import preprocess_sqr

def main():
    data_path = "../../data/"

    #Quora question pairs
    train, valid, test = preprocess_sqr.\
        transform_quora_question_pairs_to_duplicates_dfs(data_path+"qqp/train.csv")
    train.to_csv(data_path+"qqp/train.tsv", sep="\t", index=False)
    valid.to_csv(data_path+"qqp/valid.tsv", sep="\t", index=False)
    test.to_csv(data_path+"qqp/test.tsv", sep="\t", index=False)

    train, valid, test = preprocess_sqr.\
        transform_linkso_to_duplicates_dfs(data_path+"linkso/topublish/")
    train.to_csv(data_path+"linkso/train.tsv", sep="\t", index=False)
    valid.to_csv(data_path+"linkso/valid.tsv", sep="\t", index=False)
    test.to_csv(data_path+"linkso/test.tsv", sep="\t", index=False)

if __name__ == "__main__":
    main()
