import pandas as pd

def main():
    data_path = "../../data/"

    train = pd.read_csv(data_path+"clariq/train.tsv", sep="\t")
    dev = pd.read_csv(data_path+"clariq/dev.tsv", sep="\t")

    train = train[["initial_request", "question"]]
    train.columns = ["query", "clarifying_question"]
    train = train[~train["clarifying_question"].isnull()]

    dev = dev[["initial_request", "question"]]
    dev.columns = ["query", "clarifying_question"]
    dev = dev[~dev["clarifying_question"].isnull()]

    valid, test = dev[:dev.shape[0]//2], dev[dev.shape[0]//2:]

    train.to_csv(data_path+"clariq/train.tsv", sep="\t", index=False)
    valid.to_csv(data_path+"clariq/valid.tsv", sep="\t", index=False)
    test.to_csv(data_path+"clariq/test.tsv", sep="\t", index=False)

if __name__ == "__main__":
    main()