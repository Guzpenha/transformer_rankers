from os import listdir
from os.path import isfile, join
from IPython import embed
from scipy.stats import ttest_rel

import pandas as pd
import argparse
import logging


def main():

    onlyfiles = [f for f in listdir("../../data/output_data/") if isfile(join("../../data/output_data/", f)) and 'recall_df' in f]    
    dfs = []
    for f_name in onlyfiles:
        df = pd.read_csv(join("../../data/output_data/", f_name), sep="\t")
        if 'ubuntu' in f_name:
            df['task'] = 'ubuntu'
            df['method'] = f_name.replace("recall_df", "").replace("ubuntu_dstc8", "").replace(".csv", "")
        if 'mantis' in f_name:
            df['task'] = 'mantis'
            df['method'] = f_name.replace("recall_df", "").replace("mantis", "").replace(".csv", "")
        if 'msdialog' in f_name:
            df['task'] = 'msdialog'
            df['method'] = f_name.replace("recall_df", "").replace("msdialog", "").replace(".csv", "")
        dfs.append(df)
    df_all = pd.concat(dfs)
    print(df_all['method'].unique())
    print(df_all.groupby(['task', 'method']).mean())
    
    N=14
    for task in ['mantis', 'msdialog', 'ubuntu']:
        res = []
        # print("BM25 vs BM25+RM3")
        aux = ['1a vs 1b']
        df_slice_base = df_all[(df_all["task"] == task) & (df_all["method"] == "_bm25_")]
        df_slice_comparison = df_all[(df_all["task"] == task) & (df_all["method"] == "_bm25rm3_")]
        # print("R@10")
        stat, pvalue = ttest_rel(df_slice_base['R@10'].values.tolist(), df_slice_comparison['R@10'].values.tolist())
        # print(pvalue<(0.05/N))
        aux.append(pvalue<(0.05/N))
        # print("R@1")
        stat, pvalue = ttest_rel(df_slice_base['R@1'].values.tolist(), df_slice_comparison['R@1'].values.tolist())
        # print(pvalue<(0.05/N))
        aux.append(pvalue<(0.05/N))
        res.append(aux)
        
        
        # print("resp2context-lu vs bm25")
        aux = ['2b vs 1a']
        df_slice_base = df_all[(df_all["task"] == task) & (df_all["method"] == "_bm25__resp2context_last_utt_True")]
        df_slice_comparison = df_all[(df_all["task"] == task) & (df_all["method"] == "_bm25_")]
        # print("R@10")
        stat, pvalue = ttest_rel(df_slice_base['R@10'].values.tolist(), df_slice_comparison['R@10'].values.tolist())
        # print(pvalue<(0.05/N))
        aux.append(pvalue<(0.05/N))
        # print("R@1")
        stat, pvalue = ttest_rel(df_slice_base['R@1'].values.tolist(), df_slice_comparison['R@1'].values.tolist())
        # print(pvalue<(0.05/N))
        aux.append(pvalue<(0.05/N))
        res.append(aux)

        # print("resp2context vs resp2context-lu")
        aux = ['2b vs 2a']
        df_slice_base = df_all[(df_all["task"] == task) & (df_all["method"] == "_bm25__resp2context_last_utt_False")]
        df_slice_comparison = df_all[(df_all["task"] == task) & (df_all["method"] == "_bm25__resp2context_last_utt_True")]
        # print("R@10")
        stat, pvalue = ttest_rel(df_slice_base['R@10'].values.tolist(), df_slice_comparison['R@10'].values.tolist())
        # print(pvalue<(0.05/N))
        aux.append(pvalue<(0.05/N))
        # print("R@1")
        stat, pvalue = ttest_rel(df_slice_base['R@1'].values.tolist(), df_slice_comparison['R@1'].values.tolist())
        # print(pvalue<(0.05/N))
        aux.append(pvalue<(0.05/N))
        res.append(aux)

        aux = ['3e vs 1a']
        df_slice_base = df_all[(df_all["task"] == task) & (df_all["method"] == "__all-mpnet-base-v2")]
        df_slice_comparison = df_all[(df_all["task"] == task) & (df_all["method"] == "_bm25_")]
        # print("R@10")
        stat, pvalue = ttest_rel(df_slice_base['R@10'].values.tolist(), df_slice_comparison['R@10'].values.tolist())
        # print(pvalue<(0.05/N))
        aux.append(pvalue<(0.05/N))
        # print("R@1")
        stat, pvalue = ttest_rel(df_slice_base['R@1'].values.tolist(), df_slice_comparison['R@1'].values.tolist())
        # print(pvalue<(0.05/N))
        aux.append(pvalue<(0.05/N))
        res.append(aux)

        aux = ['3e vs 2b']
        df_slice_base = df_all[(df_all["task"] == task) & (df_all["method"] == "__all-mpnet-base-v2")]
        df_slice_comparison = df_all[(df_all["task"] == task) & (df_all["method"] == "_bm25__resp2context_last_utt_True")]
        # print("R@10")
        stat, pvalue = ttest_rel(df_slice_base['R@10'].values.tolist(), df_slice_comparison['R@10'].values.tolist())
        # print(pvalue<(0.05/N))
        aux.append(pvalue<(0.05/N))
        # print("R@1")
        stat, pvalue = ttest_rel(df_slice_base['R@1'].values.tolist(), df_slice_comparison['R@1'].values.tolist())
        # print(pvalue<(0.05/N))
        aux.append(pvalue<(0.05/N))
        res.append(aux)


        aux = ['3e']
        df_slice_base = df_all[(df_all["task"] == task) & (df_all["method"] == "__all-mpnet-base-v2")]
        df_slice_comparison = df_all[(df_all["task"] == task) & (df_all["method"] == "__msmarco-roberta-base-ance-firstp")]
        # print("R@10")
        stat, pvalue = ttest_rel(df_slice_base['R@10'].values.tolist(), df_slice_comparison['R@10'].values.tolist())
        # print(pvalue<(0.05/N))
        aux.append(pvalue<(0.05/N))
        # print("R@1")
        stat, pvalue = ttest_rel(df_slice_base['R@1'].values.tolist(), df_slice_comparison['R@1'].values.tolist())
        # print(pvalue<(0.05/N))
        aux.append(pvalue<(0.05/N))
        res.append(aux)

        aux = ['3e']
        df_slice_base = df_all[(df_all["task"] == task) & (df_all["method"] == "__all-mpnet-base-v2")]
        df_slice_comparison = df_all[(df_all["task"] == task) & (df_all["method"] == "__msmarco-distilbert-base-tas-b")]
        # print("R@10")
        stat, pvalue = ttest_rel(df_slice_base['R@10'].values.tolist(), df_slice_comparison['R@10'].values.tolist())
        # print(pvalue<(0.05/N))
        aux.append(pvalue<(0.05/N))
        # print("R@1")
        stat, pvalue = ttest_rel(df_slice_base['R@1'].values.tolist(), df_slice_comparison['R@1'].values.tolist())
        # print(pvalue<(0.05/N))
        aux.append(pvalue<(0.05/N))
        res.append(aux)

        aux = ['3e']
        df_slice_base = df_all[(df_all["task"] == task) & (df_all["method"] == "__all-mpnet-base-v2")]
        df_slice_comparison = df_all[(df_all["task"] == task) & (df_all["method"] == "__msmarco-bert-base-dot-v5")]
        # print("R@10")
        stat, pvalue = ttest_rel(df_slice_base['R@10'].values.tolist(), df_slice_comparison['R@10'].values.tolist())
        # print(pvalue<(0.05/N))
        aux.append(pvalue<(0.05/N))
        # print("R@1")
        stat, pvalue = ttest_rel(df_slice_base['R@1'].values.tolist(), df_slice_comparison['R@1'].values.tolist())
        # print(pvalue<(0.05/N))
        aux.append(pvalue<(0.05/N))
        res.append(aux)

        aux = ['3e']
        df_slice_base = df_all[(df_all["task"] == task) & (df_all["method"] == "__all-mpnet-base-v2")]
        df_slice_comparison = df_all[(df_all["task"] == task) & (df_all["method"] == "__multi-qa-mpnet-base-dot-v1")]
        # print("R@10")
        stat, pvalue = ttest_rel(df_slice_base['R@10'].values.tolist(), df_slice_comparison['R@10'].values.tolist())
        # print(pvalue<(0.05/N))
        aux.append(pvalue<(0.05/N))
        # print("R@1")
        stat, pvalue = ttest_rel(df_slice_base['R@1'].values.tolist(), df_slice_comparison['R@1'].values.tolist())
        # print(pvalue<(0.05/N))
        aux.append(pvalue<(0.05/N))
        res.append(aux)


        aux = ['4a vs 1a']
        df_slice_base = df_all[(df_all["task"] == task) & (df_all["method"] == "_all-mpnet-base-v2__ns_random_loss_MultipleNegativesRankingLoss")]
        df_slice_comparison = df_all[(df_all["task"] == task) & (df_all["method"] == "_bm25_")]
        # print("R@10")
        stat, pvalue = ttest_rel(df_slice_base['R@10'].values.tolist(), df_slice_comparison['R@10'].values.tolist())
        # print(pvalue<(0.05/N))
        aux.append(pvalue<(0.05/N))
        # print("R@1")
        stat, pvalue = ttest_rel(df_slice_base['R@1'].values.tolist(), df_slice_comparison['R@1'].values.tolist())
        # print(pvalue<(0.05/N))
        aux.append(pvalue<(0.05/N))
        res.append(aux)

        aux = ['4a vs 2b']
        df_slice_base = df_all[(df_all["task"] == task) & (df_all["method"] == "_all-mpnet-base-v2__ns_random_loss_MultipleNegativesRankingLoss")]
        df_slice_comparison = df_all[(df_all["task"] == task) & (df_all["method"] == "_bm25__resp2context_last_utt_True")]
        # print("R@10")
        stat, pvalue = ttest_rel(df_slice_base['R@10'].values.tolist(), df_slice_comparison['R@10'].values.tolist())
        # print(pvalue<(0.05/N))
        aux.append(pvalue<(0.05/N))
        # print("R@1")
        stat, pvalue = ttest_rel(df_slice_base['R@1'].values.tolist(), df_slice_comparison['R@1'].values.tolist())
        # print(pvalue<(0.05/N))
        aux.append(pvalue<(0.05/N))
        res.append(aux)

        aux = ['4a vs 3e']
        df_slice_base = df_all[(df_all["task"] == task) & (df_all["method"] == "_all-mpnet-base-v2__ns_random_loss_MultipleNegativesRankingLoss")]
        df_slice_comparison = df_all[(df_all["task"] == task) & (df_all["method"] == "__all-mpnet-base-v2")]
        # print("R@10")
        stat, pvalue = ttest_rel(df_slice_base['R@10'].values.tolist(), df_slice_comparison['R@10'].values.tolist())
        # print(pvalue<(0.05/N))
        aux.append(pvalue<(0.05/N))
        # print("R@1")
        stat, pvalue = ttest_rel(df_slice_base['R@1'].values.tolist(), df_slice_comparison['R@1'].values.tolist())
        # print(pvalue<(0.05/N))
        aux.append(pvalue<(0.05/N))
        res.append(aux)


        aux = ['4a vs 4b']
        df_slice_base = df_all[(df_all["task"] == task) & (df_all["method"] == "_all-mpnet-base-v2__ns_random_loss_MultipleNegativesRankingLoss")]
        df_slice_comparison = df_all[(df_all["task"] == task) & (df_all["method"] == "_all-mpnet-base-v2__ns_bm25_loss_MultipleNegativesRankingLoss")]
        # print("R@10")
        stat, pvalue = ttest_rel(df_slice_base['R@10'].values.tolist(), df_slice_comparison['R@10'].values.tolist())
        # print(pvalue<(0.05/N))
        aux.append(pvalue<(0.05/N))
        # print("R@1")
        stat, pvalue = ttest_rel(df_slice_base['R@1'].values.tolist(), df_slice_comparison['R@1'].values.tolist())
        # print(pvalue<(0.05/N))
        aux.append(pvalue<(0.05/N))
        res.append(aux)

        aux = ['4a vs 4c']
        df_slice_base = df_all[(df_all["task"] == task) & (df_all["method"] == "_all-mpnet-base-v2__ns_random_loss_MultipleNegativesRankingLoss")]
        df_slice_comparison = df_all[(df_all["task"] == task) & (df_all["method"] == "_all-mpnet-base-v2__ns_sentence_transformer_all-mpnet-base-v2_loss_MultipleNegativesRankingLoss")]
        # print("R@10")
        stat, pvalue = ttest_rel(df_slice_base['R@10'].values.tolist(), df_slice_comparison['R@10'].values.tolist())
        # print(pvalue<(0.05/N))
        aux.append(pvalue<(0.05/N))
        # print("R@1")
        stat, pvalue = ttest_rel(df_slice_base['R@1'].values.tolist(), df_slice_comparison['R@1'].values.tolist())
        # print(pvalue<(0.05/N))
        aux.append(pvalue<(0.05/N))
        res.append(aux)

        print(task)
        print(pd.DataFrame(res)[[0,2,1]])

    embed()

if __name__ == "__main__":
    main()