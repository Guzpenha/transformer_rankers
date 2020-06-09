from IPython import embed
import pandas as pd
import json

def read_crr_tsv_as_df(path, nrows=-1):
    with open(path, 'r') as f:
        df = []
        for idx, l in enumerate(f):
            if nrows != -1 and idx>nrows:
                break
            splitted = l.split("\t")
            label, utterances, candidate = splitted[0], splitted[1:-1], splitted[-1]            
            if label == "1":
                context = ""
                for idx, utterance in enumerate(utterances):                    
                    if (idx+1) % 2 == 0:
                        context+= utterance + " [TURN_SEP] "
                    else:
                        context+= utterance + " [UTTERANCE_SEP] "                    
                df.append([context, candidate])
    return pd.DataFrame(df, columns=["context", "response"])


def transform_dstc8_to_tsv(path):
    tsv_only_relevant = []
    with open(path) as json_file:
        data = json.load(json_file)
        for example in data:
            if len(example["options-for-correct-answers"])>0:
                tsv_instance = "1\t".join([e["utterance"] for e in example["messages-so-far"]])+"\t"+ \
                    example["options-for-correct-answers"][0]["utterance"]
                tsv_only_relevant.append(tsv_instance)
    return tsv_only_relevant