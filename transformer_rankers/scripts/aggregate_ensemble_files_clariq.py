from IPython import embed
import logging
import argparse
import pandas as pd
import os
import json

import functools 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_output_folder", default=None,
                        type=str, required=True, help="the folder containing predictions_with_dropout.csv and uncertainties.csv")    
    args = parser.parse_args()
    folders = [args.model_output_folder+"/"+name for name in os.listdir(args.model_output_folder) if os.path.isdir(args.model_output_folder+"/"+ name)]
    ex_ids = [int(f.split("/")[-1]) for f in folders]
    sorted_folders = [f for _,f in sorted(zip(ex_ids,folders), key=lambda pair: pair[0])]
    ensemble_id = 0
    preds_dfs = []
    preds_dfs_softmax = []
    for seed_i, run_folder in enumerate(sorted_folders):
        print(run_folder)
        os.makedirs(args.model_output_folder+"_agg/{}".format(ensemble_id), exist_ok=True)
        df = pd.read_csv(run_folder+"/predictions.csv")
        df_softmax = pd.read_csv(run_folder+"/predictions_softmax.csv")
        df.to_csv(args.model_output_folder+"_agg/{}/predictions_with_dropout_f_pass_{}.csv".format(ensemble_id, seed_i), index=False, header=True)
        preds_dfs.append(df)
        preds_dfs_softmax.append(df_softmax)
        labels = pd.read_csv(run_folder+"/labels.csv")
        with open(run_folder+"/config.json") as f:
            config = json.load(f)['args']
            config["num_foward_prediction_passes"] = len(sorted_folders)
            sum_df = functools.reduce(lambda df_1,df_2 : df_1+df_2, preds_dfs)
            sum_df_softmax = functools.reduce(lambda df_1,df_2 : df_1+df_2, preds_dfs_softmax)
            avg_df = sum_df/len(sorted_folders)
            avg_df_softmax = sum_df_softmax/len(sorted_folders)            

    dif_squared_dfs = []
    for pred_df in preds_dfs:  
        diff_df = (pred_df-avg_df)
        dif_squared_dfs.append(diff_df * diff_df)
    var_df = functools.reduce(lambda df_1,df_2 : df_1+df_2, dif_squared_dfs)
    var_df = var_df/len(sorted_folders)
    var_df.columns = ["uncertainty_{}".format(i) for i in range(len(var_df.columns))]  
    pd.read_csv(run_folder+"/predictions.csv").to_csv(args.model_output_folder+"_agg/{}/predictions.csv".format(ensemble_id), index=False, header=True)
    pd.read_csv(run_folder+"/predictions_softmax.csv").to_csv(args.model_output_folder+"_agg/{}/predictions_softmax.csv".format(ensemble_id), index=False, header=True)
    avg_df.to_csv(args.model_output_folder+"_agg/{}/predictions_with_dropout.csv".format(ensemble_id), index=False, header=True)
    avg_df_softmax.to_csv(args.model_output_folder+"_agg/{}/predictions_with_dropout_softmax.csv".format(ensemble_id), index=False, header=True)
    var_df.to_csv(args.model_output_folder+"_agg/{}/uncertainties.csv".format(ensemble_id), index=False, header=True)
    labels.to_csv(args.model_output_folder+"_agg/{}/labels.csv".format(ensemble_id), index=False, header=True)
    with open(args.model_output_folder+"_agg/{}/config.json".format(ensemble_id), "w") as f:
        config_w = {'args':config}
        json.dump(config_w, f, indent=4)

if __name__ == "__main__":
    main()