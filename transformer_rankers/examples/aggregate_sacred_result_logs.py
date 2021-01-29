from scipy.stats import ttest_rel
from transformer_rankers.eval import results_analyses_tools
from transformer_rankers.utils import utils
from IPython import embed

import pandas as pd
import numpy as np
import scipy.stats
import argparse
import logging
import json
import traceback
import os
import sys
import math

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

METRICS = ['R_10@1', 'R_10@2', 'R_10@5', 'R_2@1', 'ndcg_cut_10', 'recip_rank', 'map']
pd.set_option('display.max_columns', None)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_outputs_folder", default=None,
                        type=str, required=True, help="the folder containing all results in sacred format.")
    parser.add_argument("--identifier_columns", default=None,
                        type=str, required=True, help="The columns that uniquely identify a model and should be aggregated, comma separated.")
    parser.add_argument("--output_folder", default=None, type=str, required=True,
                        help="the folder to write aggregated results and analyses.")
    parser.add_argument("--baseline_values", default="", type=str, required=True,
                        help="The baseline values for the columns that uniquely identify a model, comma separated.")
    args = parser.parse_args()

    identifier_cols = args.identifier_columns.split(",")
    all_metrics = []
    all_logits = []    
    folders = [args.model_outputs_folder+name for name in os.listdir(args.model_outputs_folder) if os.path.isdir(args.model_outputs_folder+name)]
    for run_folder in folders:        
        try:
            with open(run_folder+"/config.json") as f:
                config = json.load(f)['args']
                config['seed'] = str(config['seed'])
            
                logging.info("Run %s" % (run_folder))
                logging.info("Seed %s" % config['seed'])
                logging.info("Task %s" % (config['task']))

                predictions = pd.read_csv(run_folder+"/predictions.csv")
                labels = pd.read_csv(run_folder+"/labels.csv")                
                results = results_analyses_tools.evaluate(utils.from_df_to_list_without_nans(predictions),
                                                          utils.from_df_to_list_without_nans(labels))
                predictions["seed"] = str(config['seed'])
                predictions["task"] = config['task']
                for c in identifier_cols:
                    predictions[c] = config[c]
                # all_logits.append(predictions[0:10000])

                #Add metrics to the df
                metrics_results = []
                metrics_cols = []
                for metric in METRICS:
                    res = 0
                    per_q_values = []
                    for q in results['model']['eval'].keys():
                        per_q_values.append(results['model']['eval'][q][metric])
                        res += results['model']['eval'][q][metric]
                    res /= len(results['model']['eval'].keys())
                    metrics_results+= [res, per_q_values]
                    metrics_cols+= [metric, metric+'_per_query']
                    logging.info("%s: %.4f" % (metric, res))

                all_metrics.append([config[c] for c in identifier_cols] +
                                [config['task'],
                                config['seed'],
                                run_folder] + metrics_results)
        except Exception as e:
            logging.info("Error on folder {}".format(run_folder))
            logging.info(traceback.format_exception(*sys.exc_info()))
            # raise # reraises the exception
    
    all_metrics_df = pd.DataFrame(all_metrics, columns= identifier_cols 
                                                + ['dataset', 'seed', 'run']
                                                + metrics_cols)

    agg_df = all_metrics_df.groupby( ['dataset'] + identifier_cols). \
        agg(['mean', 'std', 'count']). \
        reset_index().round(4)
    col_names = ['dataset'] + identifier_cols
    count_cols = []
    for metric in METRICS:
        col_names+=[metric+"_mean", metric+"_std",
                    metric+"_count"]
        count_cols.append(metric+"_count")

    agg_df.columns = col_names
    for metric in METRICS:
        agg_df[metric+"_ci95_hi"] = agg_df.apply(lambda r, m=metric, math=math: r[m+"_mean"] + 1.96*r[m+"_std"]/
                                            math.sqrt(r[m+"_count"]) , axis=1)
        agg_df[metric+"_ci95_lo"] = agg_df.apply(lambda r, m=metric, math=math: r[m+"_mean"] - 1.96*r[m+"_std"]/
                                            math.sqrt(r[m+"_count"]) , axis=1)

    agg_df = agg_df.drop(columns=count_cols[0:-1]) # removing duplicate count aggregations
    agg_df.columns = list(agg_df.columns[0:-1]) + ["count"]
    agg_df.sort_values(['dataset'] + identifier_cols + [metric+"_mean"])\
        .to_csv(args.output_folder+"_aggregated_results.csv", index=False, sep="\t")
    print(agg_df.sort_values(['dataset'] + identifier_cols + [metric+"_mean"]))

    # all_logits_df = pd.concat(all_logits)
    # all_logits_df.to_csv(args.output_folder+"_logits.csv", index=False, sep="\t")

    if args.baseline_values != "":
        #Calculating gains and statistical tests against baseline
        def average_by_query(group):
            metric_avgs = {}
            for metric in METRICS:
                by_query = []
                for i, r in group.iterrows():            
                    by_query.append(r[metric+"_per_query"])
                by_query = np.array(by_query)
                metric_avgs[metric+"_per_query_averaged"] = by_query.mean(axis=0)
            return pd.Series(metric_avgs)

        # average_over_queries
        averaged_over_queries = all_metrics_df.groupby( ['dataset'] + identifier_cols).apply(average_by_query). \
            reset_index()

        # join with baseline values
        baseline = args.baseline_values
        baseline_values = baseline.split(",")    
        baseline_runs = averaged_over_queries
        for i, id_col in enumerate(identifier_cols):        
            baseline_runs = baseline_runs[baseline_runs[id_col].astype(str) == baseline_values[i]]

        averaged_over_queries = averaged_over_queries.merge(baseline_runs, on=['dataset'])

        for metric in METRICS:
            averaged_over_queries["gain_"+metric] = averaged_over_queries.apply(lambda r, metric=metric:
            (r[metric+"_per_query_averaged_x"].mean()-r[metric+"_per_query_averaged_y"].mean())/r[metric+"_per_query_averaged_y"].mean(), axis=1)
            averaged_over_queries["p_value_stat_"+metric] = averaged_over_queries.apply(lambda r,f=ttest_rel, metric=metric:
            f(r[metric+"_per_query_averaged_x"], r[metric+"_per_query_averaged_y"]), axis=1)
            averaged_over_queries["pos_and_p_value<0.05"+metric] = averaged_over_queries.apply(lambda r, metric=metric:
                r["p_value_stat_"+metric][1]<0.05 and r["p_value_stat_"+metric][0] > 0, axis=1)
            averaged_over_queries["neg_and_p_value<0.05"+metric] = averaged_over_queries.apply(lambda r, metric=metric:
                r["p_value_stat_"+metric][1]<0.05 and r["p_value_stat_"+metric][0] < 0, axis=1)
        cols_to_add = [c for c in averaged_over_queries.columns if "gain_" in c]
        cols_to_add+= [c for c in averaged_over_queries.columns if "p_value" in c]        
        averaged_over_queries[['dataset'] + [i+"_x" for i in identifier_cols] + cols_to_add].round(4).\
            to_csv(args.output_folder+"_aggregated_results_gains_t_tests.csv", index=False, sep="\t")

if __name__ == "__main__":
    main()