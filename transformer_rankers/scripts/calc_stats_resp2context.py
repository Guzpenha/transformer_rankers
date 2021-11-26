from IPython import embed
from tqdm import tqdm
import pandas as pd

def main():

    def percentage_new(r):
        response = r['response'].split("[AUG]")[0]
        augmentation = ' '.join(r['response'].split("[AUG]")[1])
        resp_set = set(response.split(" "))
        new = 0
        old = 0
        for word in augmentation.split(" "):
            if word in resp_set:
                old+=1
            else:
                new+=1
        return new/float(len(augmentation.split(" ")))

    for task in ['mantis', 'msdialog', 'ubuntu_dstc8']:
        for split in ['train']: #['train', 'test']:
            df = pd.read_csv("../../data/{}_resp2context_last_utt_False/{}.tsv".format(task, split), sep='\t')
            df["len_context"] = df.apply(lambda r: len(r['context'].split(" ")), axis=1)
            df["len_response_aug"] = df.apply(lambda r: len(r['response'].split(" ")), axis=1)
            df["len_response"] = df.apply(lambda r: len(r['response'].split("[AUG]")[0].split(" ")), axis=1)
            df['percentage_new'] = df.apply(lambda r,f=percentage_new: f(r),axis=1)
            df_lu = pd.read_csv("../../data/{}_resp2context_last_utt_True/{}.tsv".format(task, split), sep='\t')
            df_lu["len_response_aug"] = df_lu.apply(lambda r: len(r['response'].split(" ")), axis=1)
            df_lu['percentage_new'] = df_lu.apply(lambda r,f=percentage_new: f(r),axis=1)
            print(task)
            print(df["len_context"].mean())
            print(df["len_response"].mean())
            print(df["len_response_aug"].mean())
            print(df_lu["len_response_aug"].mean())
            print(df['percentage_new'].mean())
            print(df_lu['percentage_new'].mean())
if __name__ == "__main__":
    main()