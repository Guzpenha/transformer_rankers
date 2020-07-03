import pandas as pd
from IPython import embed

def transform_quora_question_pairs_to_duplicates_dfs(path):
    """
    Transforms quora question pairs train.csv to train, test and valid pandas DF ["Q1", "Q2"]
    containing only duplicated questions.

    Args:
        path: str with the path for the csv file.
        
    Returns: (train, valid, test) pandas DataFrames
    """
    df = pd.read_csv(path)
    df_only_duplicates = df[df["is_duplicate"] == True][["question1", "question2"]]
    total = df_only_duplicates.shape[0]
    train_df, valid_df, test_df = (df_only_duplicates[:int(total*0.8)], \
        df_only_duplicates[int(total*0.8):int(total*0.9)], \
        df_only_duplicates[int(total*0.9):])
    return train_df, valid_df, test_df


def transform_linkso_to_duplicates_dfs(path):
    """
    Transforms linkso files to train, test and valid pandas DF ["Q1", "Q2"]
    containing only duplicated questions. Since the list of test and valid ids
    provided by the authors is identical (https://sites.google.com/view/linkso)
    we just split valid into two.

    Args:
        path: str with the folder containing java javascript python folders each containing
        the following files [<language>_cosidf.txt, <language>_qid2all.txt, <language>_test_qid.txt, 
        <language>_train_qid.txt, <language>_valid_qid.txt].
        
    Returns: (train, valid, test) pandas DataFrames
    """
    train, valid, test = [], [], []
    for folder in ["java", "python", "javascript"]:
        id_to_doc = {}
        with open(path+folder+"/"+folder+"_qid2all.txt") as f:
            for line in f:
                doc_id = line.strip().split("\t")[0]
                doc = " ".join(line.strip().split("\t")[1:])
                id_to_doc[doc_id] = doc
        with open(path+folder+"/"+folder+"_train_qid.txt") as f:
            train_ids = f.readlines()
            train_ids = set([x.strip() for x in train_ids])
        with open(path+folder+"/"+folder+"_valid_qid.txt") as f:
            valid_ids = f.readlines()
            valid_ids = set([x.strip() for x in valid_ids])
        with open(path+folder+"/"+folder+"_test_qid.txt") as f:
            test_ids = f.readlines()
            test_ids = set([x.strip() for x in test_ids])
        labels_df = pd.read_csv(path+folder+"/"+folder+"_cosidf.txt", sep="\t")        
        labels_df_only_duplicate = labels_df[labels_df["label"] == 1]
        labels_df_only_duplicate["qid1"] = labels_df_only_duplicate["qid1"].astype(str)
        labels_df_only_duplicate["qid2"] = labels_df_only_duplicate["qid2"].astype(str)
        
        for _, r in labels_df_only_duplicate.iterrows():
            if r["qid1"] in train_ids:
                train.append([id_to_doc[r["qid1"]], id_to_doc[r["qid2"]]])
            elif r["qid1"] in valid_ids:
                valid.append([id_to_doc[r["qid1"]], id_to_doc[r["qid2"]]])
            elif r["qid1"] in test_ids:
                test.append([id_to_doc[r["qid1"]], id_to_doc[r["qid2"]]])
            else:
                print(r["qid1"])
    train_df = pd.DataFrame(train, columns=["question1", "question2"])
    valid_df = pd.DataFrame(valid[0:len(valid)//2], columns=["question1", "question2"])
    test_df = pd.DataFrame(valid[len(valid)//2:], columns=["question1", "question2"])
    return train_df, valid_df, test_df