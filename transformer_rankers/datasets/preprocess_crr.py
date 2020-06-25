from IPython import embed
import pandas as pd
import json
import re

def read_crr_tsv_as_df(path, nrows=-1, add_turn_separator=True):
    with open(path, 'r', encoding="utf-8") as f:
        df = []
        for idx, l in enumerate(f):
            if nrows != -1 and idx>nrows:
                break
            splitted = l.split("\t")
            label, utterances, candidate = splitted[0], splitted[1:-1], splitted[-1]            
            if label == "1":
                context = ""
                for idx, utterance in enumerate(utterances):                    
                        if (idx+1) % 2 == 0 and add_turn_separator:
                            context+= utterance + " [TURN_SEP] "
                        else:
                            context+= utterance + " [UTTERANCE_SEP] "
                df.append([context.strip(), candidate.strip()])
    return pd.DataFrame(df, columns=["context", "response"])


def transform_dstc8_to_tsv(path):
    def remove_participant(utterance):
        no_participant = re.sub(r'participant_\d+ : ', '', utterance)
        no_participant = re.sub(r'participant_\d+', '', no_participant)
        return no_participant

    tsv_only_relevant = []
    with open(path) as json_file:
        data = json.load(json_file)
        for example in data:
            if len(example["options-for-correct-answers"])>0:                
                tsv_instance = "1\t" + "\t".join([remove_participant(e["utterance"]) for e in example["messages-so-far"]])+"\t"+ \
                    remove_participant(example["options-for-correct-answers"][0]["utterance"]) + "\n"
                tsv_only_relevant.append(tsv_instance)
    return tsv_only_relevant