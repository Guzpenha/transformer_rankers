from IPython import embed
import pandas as pd
import pickle
import uuid
import urllib.request
import html2text

def main():
    data_path = "../../data/"
    train = pd.read_csv(data_path+"clariq/train_original.tsv", sep="\t")
    dev = pd.read_csv(data_path+"clariq/dev.tsv", sep="\t")

    train = pd.concat([train,dev])
    with open(data_path+"clariq/top10k_docs_dict.pkl", "rb") as f:
        top10k_docs = pickle.load(f)
    query_to_top_10_documents_ids = {}    
    train = train[["topic_id", "initial_request"]].drop_duplicates()
    for _, r in train.iterrows():
        query_to_top_10_documents_ids[r["initial_request"]] = [(str(uuid.uuid5(uuid.NAMESPACE_URL, d.split("-")[0] + ":" + d)), d.split("-")[0])
                                                                         for d in top10k_docs[int(r["topic_id"])]][0:10]

    h = html2text.HTML2Text()
    h.ignore_links = True
    query_to_docs = {}
    for query in [k for k in query_to_top_10_documents_ids.keys()]:
        docs = []
        for identifier, collection in query_to_top_10_documents_ids[query]:
            if collection == "clueweb09":
                index = "cw09"
            elif collection == "clueweb12":
                index = "cw12"
            contents = urllib.request.urlopen("https://www.chatnoir.eu/cache?&uuid={}&index={}&raw".format(identifier, index)).read()
            clean_contents = h.handle(str(contents)).replace('\n','').replace('\t', '').replace('\\n', '')
            docs.append(clean_contents)
        query_to_docs[query] = docs    

    with open(data_path+"clariq/query_top_10_documents.pkl", "wb") as f:
        pickle.dump(query_to_docs, f)

    # with open(data_path+"clariq/query_top_10_documents.pkl", "rb") as f:
    #     a = pickle.load(f)

    embed()

if __name__ == "__main__":
    main()