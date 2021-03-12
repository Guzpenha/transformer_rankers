from transformer_rankers.datasets import processors
from IPython import embed
import pandas as pd
import requests
import logging
import os
import wget

TASK_TO_URLS = {
    # Clarifying Question Retrieval (CQR)
    "clariq": ["https://github.com/aliannejadi/ClariQ/raw/master/data/train.tsv",
               "https://github.com/aliannejadi/ClariQ/raw/master/data/dev.tsv"], # https://github.com/aliannejadi/ClariQ
    # Conversation Response Ranking (CRR)
    "mantis": [ "https://docs.google.com/uc?export=download&id=17Uj9EwyGGCk9w_LIqDjlTx1y4MU7xxPv"], # https://guzpenha.github.io/MANtIS/ 
    "msdialog": [ "https://docs.google.com/uc?export=download&id=1R_c8b7Yi0wChA_du3eKDtnOGuYTqVhnY"], # https://ciir.cs.umass.edu/downloads/msdialog/
    "ubuntu_dstc8": ["https://docs.google.com/uc?export=download&id=1Ypu-tIu4nT3rZ86bcqAx-lKeNomyve5N"], # https://github.com/dstc8-track2/NOESIS-II
    # Passage Retrieval (PR)
    "trec2020pr": ["https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz", 
                    "https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz",
                    "https://msmarco.blob.core.windows.net/msmarcoranking/qrels.dev.tsv",
                    "https://msmarco.blob.core.windows.net/msmarcoranking/qrels.train.tsv"], #https://microsoft.github.io/TREC-2020-Deep-Learning/
    "antique": ["ir-datasets"], 
    #Similar Question Retrieval (SQR)
    "qqp": ["https://docs.google.com/uc?export=download&id=1KAFO5l7H89zuNcSQrH08JvcD5bM7S2A_"], # https://www.kaggle.com/c/quora-question-pairs
    "linkso": ["https://docs.google.com/uc?export=download&id=1X5GoVi_OcRxahXH1pRW7TSesZUeMH3ss"] # https://sites.google.com/view/linkso
}

TASK_TO_PROCESSOR = {
    # Clarifying Question Retrieval (CQR)
    "clariq": processors.clariq_processor,
    # Conversation Response Ranking (CRR)
    "mantis": processors.mantis_processor,
    "msdialog": processors.msdialog_processor,
    "ubuntu_dstc8": processors.ubuntu_dstc8_processor,
    # Passage Retrieval (PR)
    "trec2020pr": processors.trec2020pr_processor,
    "antique": processors.antique,
    #Similar Question Retrieval (SQR)
    "qqp": processors.qqp_processor,
    "linkso": processors.linkso_processor
}

class DataDownloader():
    """
    Downloads one of the datasets from the pre-defined list.
    """
    def __init__(self, task, data_folder):
        if task not in TASK_TO_URLS:
            raise Exception("Task not in the list of available tasks: [{}]".format(", ".join(TASK_TO_URLS)))

        self.task = task
        self.urls_to_download = TASK_TO_URLS[self.task]
        self.processor = TASK_TO_PROCESSOR[self.task]
        self.data_folder = data_folder

    def download_and_preprocess(self):
        os.makedirs(self.data_folder+self.task, exist_ok=True)
        for url in self.urls_to_download:
            if "ir-datasets" in url:
                logging.info("Resorting to ir-datasets downloader.")
            elif "docs.google.com" in url:
                logging.info("Downloading {}".format(url))
                download_file_from_google_drive(url.split("id=")[-1], self.data_folder+self.task+"/drive_file")
            else:
                logging.info("Downloading {}".format(url))
                wget.download(url, out=self.data_folder+self.task+"/"+url.split("/")[-1])
        logging.info("Processing files.")
        self.processor(self.data_folder+self.task+"/")


# Code to download from google drive from 
# https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)