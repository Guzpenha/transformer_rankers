from transformer_rankers.datasets import preprocess_pr, preprocess_crr, preprocess_sqr
from IPython import embed

import pandas as pd
import py7zr
import os
import tarfile
import shutil
import zipfile
import ir_datasets


def antique(data_folder):
    """
    Downloads and transforms the ANTIQUE dataset using the ir_datasets library
    """    
    def load_split(split):    
        dataset = ir_datasets.load(split)
        docs = {}
        for doc_id, text in dataset.docs_iter():
            docs[doc_id] = text

        query = {}
        for query_id, text in dataset.queries_iter():
            query[query_id] = text

        split_data = []
        for query_id, doc_id, rel, iteration in dataset.qrels_iter():
            split_data.append([query[query_id], docs[doc_id], rel])        
        return pd.DataFrame(split_data, columns=['query', 'passage', 'rel'])

    train = load_split('antique/train/split200-train')
    valid = load_split('antique/train/split200-valid')
    test = load_split('antique/test')

    train.to_csv(data_folder+"train.tsv", sep="\t", index=False)
    valid.to_csv(data_folder+"valid.tsv", sep="\t", index=False)
    test.to_csv(data_folder+"test.tsv", sep="\t", index=False)

def linkso_processor(data_folder):
    """
    Extracts the LINKSO files downloaded from the drive and creates
    dfs with ["question", "similar_question"].
    """
    # tar xvf linkso.tar.gz
    collection_tar = tarfile.open(data_folder+"drive_file")
    collection_tar.extractall(data_folder) 
    collection_tar.close()

    train, valid, test = preprocess_sqr.\
        transform_linkso_to_duplicates_dfs(data_folder+"topublish/")
    train.to_csv(data_folder+"train.tsv", sep="\t", index=False)
    valid.to_csv(data_folder+"valid.tsv", sep="\t", index=False)
    test.to_csv(data_folder+"test.tsv", sep="\t", index=False)

    os.remove(data_folder+"drive_file")
    shutil.rmtree(data_folder+"topublish")

def qqp_processor(data_folder):
    """
    Extracts the files from Quora Question Pairs downloaded from the drive 
    and creates dfs with ["question", "similar_question"].
    """
    with zipfile.ZipFile(data_folder+"drive_file","r") as zip_ref:
        zip_ref.extractall(data_folder)

    with zipfile.ZipFile(data_folder+"train.csv.zip","r") as zip_ref:
        zip_ref.extractall(data_folder)

    train, valid, test = preprocess_sqr.\
        transform_quora_question_pairs_to_duplicates_dfs(data_folder+"train.csv")
    train.to_csv(data_folder+"train.tsv", sep="\t", index=False)
    valid.to_csv(data_folder+"valid.tsv", sep="\t", index=False)
    test.to_csv(data_folder+"test.tsv", sep="\t", index=False)

    os.remove(data_folder+"drive_file")
    os.remove(data_folder+"train.csv.zip")
    os.remove(data_folder+"sample_submission.csv.zip")
    os.remove(data_folder+"test.csv.zip")
    os.remove(data_folder+"train.csv")
    os.remove(data_folder+"test.csv")

def trec2020pr_processor(data_folder):
    """
    Extracts the files downloaded and process them into a DF with ["query", "passage"]
    """
    collection_tar = tarfile.open(data_folder+"collection.tar.gz")
    collection_tar.extractall(data_folder) 
    collection_tar.close()
    queries_tar = tarfile.open(data_folder+"queries.tar.gz")
    queries_tar.extractall(data_folder)
    queries_tar.close()

    train, valid, test = preprocess_pr.\
        transform_trec2020pr_to_dfs(data_folder)
    train.to_csv(data_folder+"/train.tsv", sep="\t", index=False)
    valid.to_csv(data_folder+"/valid.tsv", sep="\t", index=False)
    test.to_csv(data_folder+"/test.tsv", sep="\t", index=False)

    os.remove(data_folder+"collection.tar.gz")
    os.remove(data_folder+"collection.tsv")
    os.remove(data_folder+"qrels.dev.tsv")
    os.remove(data_folder+"qrels.train.tsv")
    os.remove(data_folder+"queries.dev.tsv")
    os.remove(data_folder+"queries.eval.tsv")
    os.remove(data_folder+"queries.tar.gz")
    os.remove(data_folder+"queries.train.tsv")

def clariq_processor(data_folder):
    """
    Gets the train and dev files downloaded from the github and transform it into a DF with
    ["query", "clarifying_question"]
    """
    train = pd.read_csv(data_folder+"train.tsv", sep="\t")
    dev = pd.read_csv(data_folder+"dev.tsv", sep="\t")

    train = train[["initial_request", "question"]]
    train.columns = ["query", "clarifying_question"]
    train = train[~train["clarifying_question"].isnull()]

    dev = dev[["initial_request", "question"]]
    dev.columns = ["query", "clarifying_question"]
    dev = dev[~dev["clarifying_question"].isnull()]

    valid, test = dev[:dev.shape[0]//2], dev[dev.shape[0]//2:]

    train.to_csv(data_folder+"train.tsv", sep="\t", index=False)
    valid.to_csv(data_folder+"valid.tsv", sep="\t", index=False)
    test.to_csv(data_folder+"test.tsv", sep="\t", index=False)
    os.remove(data_folder+"dev.tsv")

def ubuntu_dstc8_processor(data_folder):
    """
    Gets the compresse file "drive_file" containing the ubuntu_dstc8 data
    and preprocess it into a DF with ["conversational_context", "response"] columns.
    """
    #Deal with drive files
    with zipfile.ZipFile(data_folder+"drive_file","r") as zip_ref:
        zip_ref.extractall(data_folder)

    for f_path, f_o_path in [("{}/ubuntu/task-1.ubuntu.dev.json".format(data_folder), "{}/valid.tsv".format(data_folder)),
                            ("{}/ubuntu/task-1.ubuntu.train.json".format(data_folder), "{}/train.tsv".format(data_folder))]:
        print("transforming {}".format(f_path))
        data = preprocess_crr.transform_dstc8_to_tsv(f_path)
        with open(f_o_path, 'w') as f_write:
            for l in data:
                f_write.write(l)

    #Transform tsv to two column pandas dfs.
    train = preprocess_crr.read_crr_tsv_as_df(data_folder+"/train.tsv", -1, add_turn_separator=False)
    valid = preprocess_crr.read_crr_tsv_as_df(data_folder+"/valid.tsv", -1, add_turn_separator=False)
    
    #Save files with correct formats
    train.to_csv(data_folder+"/train.tsv", index=False, sep="\t")
    valid[0:int(valid.shape[0]/2)].to_csv(data_folder+"/valid.tsv", index=False, sep="\t")
    valid[int(valid.shape[0]/2):].to_csv(data_folder+"/test.tsv", index=False, sep="\t")

    #Clean folder
    os.remove(data_folder+"drive_file")
    os.remove(data_folder+"task-1.ubuntu.test.blind.json")
    os.remove(data_folder+"ubuntu-task-1.txt")
    shutil.rmtree(data_folder+"ubuntu")

def msdialog_processor(data_folder):
    """
    Gets the compresse file "drive_file" containing the msdialog data
    and preprocess it into a DF with ["conversational_context", "response"] columns.
    """
    #Deal with drive files
    my_tar = tarfile.open(data_folder+"drive_file")
    my_tar.extractall(data_folder) 
    my_tar.close()

    #Transform tsv to two column pandas dfs.
    train = preprocess_crr.read_crr_tsv_as_df(data_folder+"MSDialog/train.tsv")
    valid = preprocess_crr.read_crr_tsv_as_df(data_folder+"MSDialog/valid.tsv")
    test = preprocess_crr.read_crr_tsv_as_df(data_folder+"MSDialog/test.tsv")

    #Save files with correct formats
    train.to_csv(data_folder+"/train.tsv", index=False, sep="\t")
    valid.to_csv(data_folder+"/valid.tsv", index=False, sep="\t")
    test.to_csv(data_folder+"/test.tsv", index=False, sep="\t")

    #Clean folder
    os.remove(data_folder+"drive_file")
    shutil.rmtree(data_folder+"MSDialog")

def mantis_processor(data_folder):
    """
    Gets the compresse file "drive_file" containing the mantis data
    and preprocess it into a DF with ["conversational_context", "response"] columns.
    """
    #Deal with drive files
    archive = py7zr.SevenZipFile(data_folder+"drive_file", mode='r')
    archive.extractall(path=data_folder)
    archive.close()

    #Transform tsv to two column pandas dfs.
    train = preprocess_crr.read_crr_tsv_as_df(data_folder+"/data_train_easy.tsv")
    valid = preprocess_crr.read_crr_tsv_as_df(data_folder+"/data_dev_easy.tsv")
    test =  preprocess_crr.read_crr_tsv_as_df(data_folder+"/data_test_easy.tsv")

    #Save files with correct formats
    train.to_csv(data_folder+"/train.tsv", index=False, sep="\t")
    valid.to_csv(data_folder+"/valid.tsv", index=False, sep="\t")
    test.to_csv(data_folder+"/test.tsv", index=False, sep="\t")

    #Clean folder
    os.remove(data_folder+"drive_file")
    os.remove(data_folder+"data_dev_easy_lookup.txt")
    os.remove(data_folder+"data_test_easy_lookup.txt")
    os.remove(data_folder+"data_train_easy_lookup.txt")
    os.remove(data_folder+"data_train_easy.tsv")
    os.remove(data_folder+"data_dev_easy.tsv")
    os.remove(data_folder+"data_test_easy.tsv")

