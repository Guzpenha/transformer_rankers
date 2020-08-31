from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyClassifier
from transformer_rankers.eval import results_analyses_tools
from transformer_rankers.utils import utils
from statistics import mean, median, stdev, variance
from IPython import embed

from scipy.stats import logistic, ttest_rel
import numpy as np
import pandas as pd
import logging
import argparse
import random
import json
import os

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

    eval_metric='R_10@1'
    folders = [args.model_output_folder+name for name in os.listdir(args.model_output_folder) if os.path.isdir(args.model_output_folder+name)]
    all_nota_dfs = []
    all_risk_aware_dfs = []
    all_reliability_diagrams = []
    eces_df = []
    print(folders)
    for run_folder in folders:
        uncertainties = pd.read_csv(run_folder+"/uncertainties.csv")
        point_estimate_predictions = pd.read_csv(run_folder+"/predictions.csv")
        sthocastic_predictions = pd.read_csv(run_folder+"/predictions_with_dropout.csv")
        sthocastic_predictions_softmax = pd.read_csv(run_folder+"/predictions_with_dropout_softmax.csv")
        point_estimate_predictions_softmax = pd.read_csv(run_folder+"/predictions_softmax.csv")
        labels = pd.read_csv(run_folder+"/labels.csv")

        with open(run_folder+"/config.json") as f:
            config = json.load(f)['args']
        logging.info("Folder: {}".format(run_folder))
        logging.info("Task: {}".format(config["task"]))
        logging.info("NS train: {}".format(config["train_negative_sampler"]))
        logging.info("NS test: {}".format(config["test_negative_sampler"]))
        logging.info("Test Task: {}".format(config["test_dataset"]))

        do_calibration_eval=True
        if do_calibration_eval:
            for preds_name, preds_for_calibration in [("point_estimate_softmax", point_estimate_predictions_softmax),
                                                    ("stochastic_softmax", sthocastic_predictions_softmax)]:
                preds_for_calibration = preds_for_calibration.loc[:, ~preds_for_calibration.isna().any()]
                labels_for_calibration = labels.loc[:, ~labels.isna().any()]
                all_res =  results_analyses_tools.evaluate_and_aggregate(utils.from_df_to_list_without_nans(preds_for_calibration),
                                                                         utils.from_df_to_list_without_nans(labels_for_calibration), metrics=['R_10@1'])
                pred_cols = preds_for_calibration.columns                
                for n_candidate_documents in range(2, 10):
                    logging.info("Using {} candidate documents".format(n_candidate_documents))
                    q_doc_per_line = []
                    for i, r in preds_for_calibration.iterrows():
                        for c_index in range(len(pred_cols[0:n_candidate_documents])):
                            pred = r[pred_cols[c_index]]
                            label = labels_for_calibration.iloc[i, c_index]
                            acc = label
                            q_doc_per_line.append([pred, label, acc])
                    q_doc_df = pd.DataFrame(q_doc_per_line, columns=["conf","label","acc"])
                    q_doc_df["errors"] = abs(q_doc_df["acc"] - q_doc_df["conf"])
                    total_size = q_doc_df.shape[0]
                    M = 50
                    bucket_size = 1.0/M
                    conf_start = 0
                    ece = 0
                    reliability_diagram = []  
                    errors = []                  
                    for _ in range(M):
                        conf_end = conf_start+bucket_size
                        df_cut = q_doc_df[(q_doc_df["conf"] >= conf_start) & (q_doc_df["conf"] < conf_end)]
                        
                        conf = np.mean(df_cut["conf"].values)

                        acc = np.mean(df_cut["acc"].values)

                        size_bucket = df_cut.shape[0]

                        if str(conf) != "nan" and str(acc) != "nan":                            
                            ece += abs(acc - conf) * (size_bucket/total_size)
                            errors.append(abs(acc - conf) * (size_bucket/total_size))
                        else:
                            print("nana")
                            errors.append(0)
                        reliability_diagram.append([config["task"], config["test_dataset"], preds_name, "train_"+config["train_negative_sampler"], 
                                                    "test_"+config["test_negative_sampler"], conf_start, conf, acc, size_bucket, n_candidate_documents])
                        conf_start+=bucket_size
                    ece = ece/M
                    logging.info(preds_name)
                    logging.info("ECE: %0.4f" % (ece))
                    eces_df.append([config["task"], config["test_dataset"], preds_name, "train_"+config["train_negative_sampler"], "test_"+config["test_negative_sampler"], ece, n_candidate_documents, all_res['R_10@1'], errors])
                    reliability_diagram_df = pd.DataFrame(reliability_diagram, columns = ["train_task", "test_task", "prediction_type", "NS_train", "NS_test","confidence_start", "conf", "acc", "size_bucket", "n_candidate_documents"])
                    all_reliability_diagrams.append(reliability_diagram_df)

        do_risk_aware_eval=True
        if do_risk_aware_eval:
            sthocastic_predictions = pd.read_csv(run_folder+"/predictions_with_dropout.csv")
            point_estimate_predictions = pd.read_csv(run_folder+"/predictions.csv")
            risk_aware_res = []
            results = results_analyses_tools.evaluate_and_aggregate(utils.from_df_to_list_without_nans(point_estimate_predictions),
                                                                    utils.from_df_to_list_without_nans(labels),[eval_metric])
            for metric in results:
                logging.info("Point estimate %s: %0.4f" % (metric, results[metric]))
            point_estimate_res = results[metric]
            # risk_aware_res.append([config["task"], "train_"+config["train_negative_sampler"], 
            #                       "test_"+config["test_negative_sampler"], "point_estimate" , results[metric], results[metric] - point_estimate_res])            
            # results = results_analyses_tools.evaluate_and_aggregate(utils.from_df_to_list_without_nans(predictions),
            #                                                         utils.from_df_to_list_without_nans(labels),[eval_metric])
            # for metric in results:
            #     logging.info("Dropout mean %s: %0.4f" % (metric, results[metric]))
            
            # risk_aware_res.append([config["task"], "train_"+config["train_negative_sampler"], 
            #                       "test_"+config["test_negative_sampler"], "MC_dropout_mean" , results[metric], results[metric] - point_estimate_res])

            def risk_aware_prediction(row, col, b, w, cov):
                document_rank = int(col.split("_")[-1])
                mean_term = row[col]            
                var_term = (b * w[document_rank] * row["uncertainty_{}".format(document_rank)])
                cov_term = 0
                if (str(mean_term) == 'nan'):
                    return 'nan'# some documents lists are bigger than others due to multiple relevant cases.
                for i in range(cov.shape[0]):
                    if i != document_rank:
                        cov_term+=w[i] * cov[document_rank][i]
                cov_term = (2*b*cov_term)

                return mean_term - var_term - cov_term

            n_f_passes = config["num_foward_prediction_passes"]
            dfs_f_passes = []
            for i in range(n_f_passes):
                f_pass_pred_df = pd.read_csv(run_folder+"/predictions_with_dropout_f_pass_{}.csv".format(i))
                dfs_f_passes.append(f_pass_pred_df)

            preds_by_query = {}
            for index, row in dfs_f_passes[0].iterrows():
                preds_by_query[index] = []
                for df in dfs_f_passes:
                    preds_by_query[index].append([v for v in df.iloc[index].tolist() if str(v) != 'nan'])
            cov_by_query = {}
            for index in preds_by_query:
                cov_by_query[index] =  np.cov(np.array(preds_by_query[index]).T, bias=True)
            
            w = [1] * sthocastic_predictions.shape[1]
            # for b in [-6,-5,-4,-3,-2,-1,1,2,3,4,5,6]:
            # for b in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]:
            # for b in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
            # for b in [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            for b in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                preds_with_var = sthocastic_predictions.join(uncertainties)
                for column in preds_with_var.columns:
                    if "prediction" in column: 
                        preds_with_var[column] = preds_with_var.apply(lambda r, f=risk_aware_prediction,
                            c=column, b=b, w=w,cov=cov_by_query : f(r,c,b,w,cov[r.name]), axis=1)
                preds_with_var = preds_with_var.drop(["uncertainty_{}".format(i) for i in range(uncertainties.shape[1])], axis=1)

                results_per_query = results = results_analyses_tools.evaluate(utils.from_df_to_list_without_nans(preds_with_var),
                                                                        utils.from_df_to_list_without_nans(labels))
                per_q_res=[]
                for query in results_per_query['model']['eval']:
                    per_q_res.append(results_per_query['model']['eval'][query][eval_metric])

                results = results_analyses_tools.evaluate_and_aggregate(utils.from_df_to_list_without_nans(preds_with_var),
                                                                        utils.from_df_to_list_without_nans(labels),[eval_metric])
                for metric in results:
                    logging.info("Risk Aware (b=%0.1f) %s: %0.4f" % (b, metric, results[metric]))
                risk_aware_res.append([config["task"], config["test_dataset"], "train_"+config["train_negative_sampler"], 
                                  "test_"+config["test_negative_sampler"], b, results[metric], results[metric] - point_estimate_res, per_q_res])
            risk_aware_res_df = pd.DataFrame(risk_aware_res, columns = ["train_task", "test_task", "NS_train", "NS_test", "b", eval_metric, "delta_point_estimate", "per_q_res"])
            all_risk_aware_dfs.append(risk_aware_res_df)
            # embed()        

        do_NOTA_experiment=True
        if do_NOTA_experiment:
            #for simplicity remove instances with multiple relevants:
            while sthocastic_predictions[sthocastic_predictions[sthocastic_predictions.columns[-1]].isnull()].shape[0] != 0:
                sthocastic_predictions = sthocastic_predictions[sthocastic_predictions[sthocastic_predictions.columns[-1]].isnull()]
                uncertainties = uncertainties[uncertainties[uncertainties.columns[-1]].isnull()]
                point_estimate_predictions = point_estimate_predictions[point_estimate_predictions[point_estimate_predictions.columns[-1]].isnull()]
                logging.info("Dropping columns: {}".format(sthocastic_predictions.columns[-1]))
                sthocastic_predictions = sthocastic_predictions.drop(sthocastic_predictions.columns[-1], axis=1)
                uncertainties = uncertainties.drop(uncertainties.columns[-1], axis=1)
                point_estimate_predictions = point_estimate_predictions.drop(point_estimate_predictions.columns[-1], axis=1)

            #create simulated NOTA dataset
            sthocastic_predictions["is_NOTA"] = False
            uncertainties["is_NOTA"] = False
            point_estimate_predictions["is_NOTA"] = False

            nota_true_pos = sthocastic_predictions.sample(int(0.5 * sthocastic_predictions.shape[0])).index
            sthocastic_predictions.loc[nota_true_pos,'is_NOTA'] = True
            uncertainties.loc[nota_true_pos,'is_NOTA'] = True
            point_estimate_predictions.loc[nota_true_pos,'is_NOTA'] = True

            def cross_validate(df, column, clf, shuffle_list=False, sorted_list=False, stats_summaries=False):
                X = []
                y = []
                for idx, r in df.iterrows():
                    label = r["is_NOTA"]
                    if label:
                        features = [r["{}_{}".format(column, i+1)] for i in range(df.shape[1]-2)]
                    else:
                        features = [r["{}_{}".format(column, i)] for i in range(df.shape[1]-2)]
                    if shuffle_list:
                        random.shuffle(features)
                    if sorted_list:
                        features = sorted(features, reverse=True)
                    if stats_summaries:                        
                        features = [max(features), min(features), np.mean(features), median(features), stdev(features), variance(features)]

                    X.append(features)
                    label = 1 if r["is_NOTA"] else 0
                    y.append(label)

                X = np.array(X)
                y = np.array(y)        

                scores = cross_val_score(clf, X, y, cv=5, scoring='f1_macro')
                logging.info("CV using {}.".format(column))
                logging.info("F1: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
                return  scores.mean(), scores.std() * 2, scores

            both = sthocastic_predictions.drop(["is_NOTA"], axis=1).join(uncertainties)
            both.columns = ["both_{}".format(i) for i in range(both.shape[1]-1)] + ["is_NOTA"]

            clf = RandomForestClassifier(max_depth=2, random_state=0)
            results = []
            logging.info("Sorted lists")
            mean_predictions_only, std , predictions_only_scores = cross_validate(sthocastic_predictions, "prediction", clf, sorted_list=True)
            results.append([config["task"], config["test_dataset"],  config["train_negative_sampler"], config["test_negative_sampler"], "prediction_only", "sorted_list", 0.0, mean_predictions_only, std, False])

            mean, std, both_scores = cross_validate(both, "both", clf, sorted_list=True)            
            print(both_scores, predictions_only_scores)
            stat, pvalue = ttest_rel(both_scores, predictions_only_scores)
            print(pvalue, stat)
            significant = pvalue/2 < 0.05 and stat > 0
            results.append([config["task"], config["test_dataset"],  config["train_negative_sampler"], config["test_negative_sampler"], "both", "sorted_list", mean-mean_predictions_only, mean, std, significant])

            results_df = pd.DataFrame(results, columns = ["train_task", "test_task", "NS_train", "NS_test", "input_space", "input_format", "f1delta", "f1", "std", "significant"])
            all_nota_dfs.append(results_df)

    if do_NOTA_experiment:
        all_nota_dfs_for_real = pd.concat(all_nota_dfs)
        dfs_for_cross_task = all_nota_dfs_for_real.drop(["input_format", "f1delta", "NS_train", "significant"], axis=1)
        dfs_for_cross_task = dfs_for_cross_task[dfs_for_cross_task["NS_test"]=="bm25"].drop(["NS_test"], axis=1)
        dfs_for_cross_task = dfs_for_cross_task.set_index(['train_task', 'test_task', 'input_space'], drop=True).unstack().unstack() 
        dfs_for_cross_task.columns = dfs_for_cross_task.columns.swaplevel(0, 2)
        dfs_for_cross_task.columns = dfs_for_cross_task.columns.swaplevel(1, 2)
        correct_order = [
            (      'mantis',  'f1', 'prediction_only'),
            (      'mantis', 'std', 'prediction_only'),
            (      'mantis',  'f1',            'both'),
            (      'mantis', 'std',            'both'),
            (    'msdialog',  'f1', 'prediction_only'),
            (    'msdialog', 'std', 'prediction_only'),
            (    'msdialog',  'f1',            'both'),
            (    'msdialog', 'std',            'both'),
            ('ubuntu_dstc8',  'f1', 'prediction_only'),
            ('ubuntu_dstc8', 'std', 'prediction_only'),
            ('ubuntu_dstc8',  'f1',            'both'),
            ('ubuntu_dstc8', 'std',            'both')
            ]
        dfs_for_cross_task[correct_order].to_csv(args.model_output_folder+"NOTA_res_cross_task.csv", sep="\t")

        dfs_for_cross_task_ttest = all_nota_dfs_for_real.drop(["input_format", "f1delta", "NS_train", "f1", "std"], axis=1)
        dfs_for_cross_task_ttest = dfs_for_cross_task_ttest[dfs_for_cross_task_ttest["input_space"]=="both"]
        dfs_for_cross_task_ttest = dfs_for_cross_task_ttest[dfs_for_cross_task_ttest["NS_test"]=="bm25"].drop(["NS_test"], axis=1)
        dfs_for_cross_task_ttest = dfs_for_cross_task_ttest.set_index(['train_task', 'test_task', 'input_space'], drop=True).unstack().unstack() 
        print(dfs_for_cross_task_ttest)

        dfs_for_cross_ns = all_nota_dfs_for_real.drop(["input_format", "f1delta", "NS_train", "significant"], axis=1)
        dfs_for_cross_ns = dfs_for_cross_ns[dfs_for_cross_ns["train_task"] == dfs_for_cross_ns["test_task"]]
        dfs_for_cross_ns = dfs_for_cross_ns[dfs_for_cross_ns["NS_test"] != "bm25"]
        dfs_for_cross_ns = dfs_for_cross_ns.drop(["test_task"], axis=1)
        dfs_for_cross_ns = dfs_for_cross_ns.set_index(['train_task', 'NS_test', 'input_space'], drop=True).unstack().unstack() 
        dfs_for_cross_ns.columns = dfs_for_cross_ns.columns.swaplevel(0, 2)
        dfs_for_cross_ns.columns = dfs_for_cross_ns.columns.swaplevel(1, 2)
        correct_order = [
            (      'random',  'f1', 'prediction_only'),
            (      'random', 'std', 'prediction_only'),
            (      'random',  'f1',            'both'),
            (      'random', 'std',            'both'),
            ('sentenceBERT',  'f1', 'prediction_only'),
            ('sentenceBERT', 'std', 'prediction_only'),
            ('sentenceBERT',  'f1',            'both'),
            ('sentenceBERT', 'std',            'both')]
        dfs_for_cross_ns[correct_order].to_csv(args.model_output_folder+"NOTA_res_cross_ns.csv", sep="\t")

        dfs_for_cross_ns_ttest = all_nota_dfs_for_real.drop(["input_format", "f1delta", "NS_train", "f1", "std"], axis=1)
        dfs_for_cross_ns_ttest = dfs_for_cross_ns_ttest[dfs_for_cross_ns_ttest["input_space"]=="both"]
        dfs_for_cross_ns_ttest = dfs_for_cross_ns_ttest[dfs_for_cross_ns_ttest["train_task"] == dfs_for_cross_ns_ttest["test_task"]]
        dfs_for_cross_ns_ttest = dfs_for_cross_ns_ttest[dfs_for_cross_ns_ttest["NS_test"]!="bm25"].drop(["test_task"], axis=1)
        dfs_for_cross_ns_ttest = dfs_for_cross_ns_ttest.set_index(['train_task', 'NS_test', 'input_space'], drop=True).unstack().unstack() 
        print(dfs_for_cross_ns_ttest)

    if do_risk_aware_eval:
        all_risk_aware_dfs = pd.concat(all_risk_aware_dfs)
        all_risk_aware_dfs.to_csv(args.model_output_folder+"RiskAware_res.csv", index=False, sep="\t")

        # embed()
        #use best b for each combination based on dev
        #calculate improvements over 0. 
        use_best_val_b=False
        if use_best_val_b:
            stochastic_means = all_risk_aware_dfs[all_risk_aware_dfs["b"] == 0].drop("delta_point_estimate", axis=1)
            risk_dfs_with_stochastic_means = all_risk_aware_dfs.merge(stochastic_means, on=["train_task", "test_task", "NS_train", "NS_test"])
            risk_dfs_with_stochastic_means.columns = ['train_task', 'test_task', 'NS_train', 'NS_test', 
                'b', 'R_10@1', 'delta_point_estimate', 'per_q_res', '_', 'R_10@1_b_zero', 'per_q_res_b_zero']
            
            dev_results = pd.read_csv("/home/guzpenha/personal/tranformer_rankers/data/output_data_uncertainty_ensemble_agg/RiskAware_res.csv", sep="\t")
            # max_idx = dev_results.groupby(["train_task", "test_task", "NS_train", "NS_test"])["R_10@1"].idxmax().values
            max_idx = dev_results[dev_results["b"]<=1.0].groupby(["train_task", "test_task", "NS_train", "NS_test"])["recip_rank"].idxmax().values

            best_b_df = risk_dfs_with_stochastic_means.merge(dev_results.iloc[max_idx], on=["train_task", "test_task", "NS_train", "NS_test" ,"b"])

            # best_b_df["percentage_improv"] = best_b_df.apply(lambda r: (r["R_10@1_x"]-r["R_10@1_b_zero"])/r["R_10@1_x"], axis=1)
            best_b_df["percentage_improv"] = best_b_df.apply(lambda r: (r["R_10@1"]-r["R_10@1_b_zero"])/r["R_10@1"], axis=1)

            dfs_for_cross_task = best_b_df
            dfs_for_cross_task = dfs_for_cross_task[dfs_for_cross_task["NS_test"]=="test_bm25"].drop(["NS_test"], axis=1)
            
            ttests_cross_task = dfs_for_cross_task.copy()
            ttests_cross_task["pvalue_stat"]  = ttests_cross_task.apply(lambda r,f=ttest_rel: f(r["per_q_res"], r["per_q_res_b_zero"]), axis=1)
            ttests_cross_task["pvalue<0.05"]  = ttests_cross_task.apply(lambda r: r["pvalue_stat"][1]/2<0.05 and r["pvalue_stat"][0]>0, axis=1)
            ttests_cross_task = ttests_cross_task.drop(['NS_train', 'b', 'R_10@1', 'recip_rank','delta_point_estimate_x', '_', 'R_10@1_b_zero', 'delta_point_estimate_y', 'per_q_res', 'per_q_res_b_zero', 'percentage_improv', 'pvalue_stat'], axis=1)
            ttests_cross_task = ttests_cross_task.set_index(['train_task', 'test_task'], drop=True).unstack()
            print(ttests_cross_task)
            
            dfs_for_cross_task = dfs_for_cross_task.drop(['NS_train', 'b', 'R_10@1', 'recip_rank','delta_point_estimate_x', '_', 'R_10@1_b_zero', 'delta_point_estimate_y', 'per_q_res', 'per_q_res_b_zero'], axis=1)
            # dfs_for_cross_task = dfs_for_cross_task.drop(['NS_train', 'b', 'R_10@1_x', 'R_10@1_y','delta_point_estimate_x', '_', 'R_10@1_b_zero', 'delta_point_estimate_y', 'per_q_res', 'per_q_res_b_zero'], axis=1)
            dfs_for_cross_task = dfs_for_cross_task.set_index(['train_task', 'test_task'], drop=True).unstack()
            dfs_for_cross_task.to_csv(args.model_output_folder+"RISK_AWARE_res_cross_task.csv", sep="\t")

            dfs_for_cross_ns = best_b_df
            dfs_for_cross_ns = dfs_for_cross_ns[dfs_for_cross_ns["train_task"]==dfs_for_cross_ns["test_task"]]
            dfs_for_cross_ns = dfs_for_cross_ns[dfs_for_cross_ns["NS_test"]!="test_bm25"]

            ttests_cross_ns = dfs_for_cross_ns.copy()
            ttests_cross_ns["pvalue_stat"]  = ttests_cross_ns.apply(lambda r,f=ttest_rel: f(r["per_q_res"], r["per_q_res_b_zero"]), axis=1)
            ttests_cross_ns["pvalue<0.05"]  = ttests_cross_ns.apply(lambda r: r["pvalue_stat"][1]/2<0.05 and r["pvalue_stat"][0]>0, axis=1)
            ttests_cross_ns = ttests_cross_ns.drop(['test_task','NS_train', 'b', 'R_10@1', 'recip_rank','delta_point_estimate_x', '_', 'R_10@1_b_zero', 'delta_point_estimate_y', 'per_q_res', 'per_q_res_b_zero', 'percentage_improv', 'pvalue_stat'], axis=1)
            ttests_cross_ns = ttests_cross_ns.set_index(['train_task', 'NS_test'], drop=True).unstack()
            print(ttests_cross_ns)

            dfs_for_cross_ns = dfs_for_cross_ns.drop(['test_task','NS_train', 'b', 'R_10@1_x', 'R_10@1_y','delta_point_estimate_x', '_', 'R_10@1_b_zero', 'delta_point_estimate_y'], axis=1)
            dfs_for_cross_ns = dfs_for_cross_ns.set_index(['train_task', 'NS_test'], drop=True).unstack()
            dfs_for_cross_ns.to_csv(args.model_output_folder+"RISK_AWARE_res_cross_ns.csv", sep="\t")

    if do_calibration_eval:
        all_reliability_diagrams_df = pd.concat(all_reliability_diagrams)
        all_reliability_diagrams_df.fillna(0.0).to_csv(args.model_output_folder+"Reliability_res.csv", index=False, sep="\t")
        eces_df = pd.DataFrame(eces_df, columns= ["train_task", "test_task", "prediction_type", "NS_train", "NS_test", "ECE", "n_candidate_documents", "R_10@1", "per_bucket_errors"])
        eces_df.to_csv(args.model_output_folder+"ECE_res.csv", index=False, sep="\t")
        n_candidate_documents = 2

        #statistical tests
        ece_and_recall = eces_df[eces_df["n_candidate_documents"]==n_candidate_documents].drop(["n_candidate_documents", "NS_train", "R_10@1", "ECE"], axis=1)
        ece_and_recall = ece_and_recall.merge(ece_and_recall[ece_and_recall["prediction_type"]=="point_estimate_softmax"], 
                                              on=["train_task", "test_task", "NS_test"])
        ece_and_recall = ece_and_recall[ece_and_recall["prediction_type_x"] == "stochastic_softmax"]
        print(ece_and_recall["per_bucket_errors_x"].apply(mean) < ece_and_recall["per_bucket_errors_y"].apply(mean))
        ece_and_recall["pvalue_stat"]  = ece_and_recall.apply(lambda r,f=ttest_rel: f(r["per_bucket_errors_x"], r["per_bucket_errors_y"]), axis=1)
        ece_and_recall["pvalue<0.05"]  = ece_and_recall.apply(lambda r: r["pvalue_stat"][1]<0.05 and r["pvalue_stat"][0]<0, axis=1)
        ece_and_recall = ece_and_recall.drop(["per_bucket_errors_x", "per_bucket_errors_y", "prediction_type_y", "pvalue_stat"], axis=1)
        ece_and_recall = ece_and_recall[ece_and_recall["NS_test"] == "test_bm25"].drop(['NS_test'],axis=1).set_index(['train_task', 'test_task', 'prediction_type_x'], drop=True).unstack().unstack() 
        print(ece_and_recall)

        ece_and_recall_cross_ns = eces_df[eces_df["n_candidate_documents"]==n_candidate_documents].drop(["n_candidate_documents", "NS_train", "R_10@1", "ECE"], axis=1)
        ece_and_recall_cross_ns = ece_and_recall_cross_ns.merge(ece_and_recall_cross_ns[ece_and_recall_cross_ns["prediction_type"]=="point_estimate_softmax"], 
                                              on=["train_task", "test_task", "NS_test"])
        ece_and_recall_cross_ns = ece_and_recall_cross_ns[ece_and_recall_cross_ns["NS_test"] != "test_bm25"].drop(["test_task"], axis=1)
        ece_and_recall_cross_ns = ece_and_recall_cross_ns[ece_and_recall_cross_ns["prediction_type_x"] == "stochastic_softmax"]
        ece_and_recall_cross_ns["pvalue_stat"]  = ece_and_recall_cross_ns.apply(lambda r,f=ttest_rel: f(r["per_bucket_errors_x"], r["per_bucket_errors_y"]), axis=1)
        ece_and_recall_cross_ns["pvalue<0.05"]  = ece_and_recall_cross_ns.apply(lambda r: r["pvalue_stat"][1]<0.05 and r["pvalue_stat"][0]<0, axis=1)        
        ece_and_recall_cross_ns = ece_and_recall_cross_ns.drop(["per_bucket_errors_x", "per_bucket_errors_y", "prediction_type_y", "pvalue_stat"], axis=1)
        ece_and_recall_cross_ns = ece_and_recall_cross_ns.set_index(['train_task', 'NS_test', 'prediction_type_x'], drop=True).unstack().unstack()
        print(ece_and_recall_cross_ns)

        #Table
        ece_and_recall = eces_df[eces_df["n_candidate_documents"]==n_candidate_documents].drop(["n_candidate_documents", "NS_train", "per_bucket_errors"], axis=1)
        ece_and_recall = ece_and_recall[ece_and_recall["NS_test"] == "test_bm25"].drop(['NS_test'],axis=1).set_index(['train_task', 'test_task', 'prediction_type'], drop=True).unstack().unstack() 
        ece_and_recall.columns = ece_and_recall.columns.swaplevel(0, 2)
        ece_and_recall.columns = ece_and_recall.columns.swaplevel(1, 2)
        correct_order = [
                        (      'mantis', 'R_10@1', 'point_estimate_softmax'),
                        (      'mantis', 'R_10@1',     'stochastic_softmax'),
                        (      'mantis',    'ECE', 'point_estimate_softmax'),
                        (      'mantis',    'ECE',     'stochastic_softmax'),
                        (    'msdialog', 'R_10@1', 'point_estimate_softmax'),
                        (    'msdialog', 'R_10@1',     'stochastic_softmax'),
                        (    'msdialog',    'ECE', 'point_estimate_softmax'),
                        (    'msdialog',    'ECE',     'stochastic_softmax'),
                        ('ubuntu_dstc8', 'R_10@1', 'point_estimate_softmax'),
                        ('ubuntu_dstc8', 'R_10@1',     'stochastic_softmax'),
                        ('ubuntu_dstc8',    'ECE', 'point_estimate_softmax'),
                        ('ubuntu_dstc8',    'ECE',     'stochastic_softmax')
                        ]
        ece_and_recall[correct_order].to_csv(args.model_output_folder+"ECE_and_Metric_cross_task.csv", sep="\t")
        correct_order = [
                        (      'test_random', 'R_10@1', 'point_estimate_softmax'),
                        (      'test_random', 'R_10@1',     'stochastic_softmax'),
                        (      'test_random',    'ECE', 'point_estimate_softmax'),
                        (      'test_random',    'ECE',     'stochastic_softmax'),
                        ('test_sentenceBERT', 'R_10@1', 'point_estimate_softmax'),
                        ('test_sentenceBERT', 'R_10@1',     'stochastic_softmax'),
                        ('test_sentenceBERT',    'ECE', 'point_estimate_softmax'),
                        ('test_sentenceBERT',    'ECE',     'stochastic_softmax')
                        ]
        ece_and_recall_cross_ns = eces_df[eces_df["n_candidate_documents"]==n_candidate_documents].drop(["n_candidate_documents", "NS_train", "per_bucket_errors"], axis=1)
        ece_and_recall_cross_ns = ece_and_recall_cross_ns[ece_and_recall_cross_ns["NS_test"] != "test_bm25"].drop(["test_task"], axis=1).set_index(['train_task', 'NS_test', 'prediction_type'], drop=True).unstack().unstack() 
        ece_and_recall_cross_ns.columns = ece_and_recall_cross_ns.columns.swaplevel(0, 2)
        ece_and_recall_cross_ns.columns = ece_and_recall_cross_ns.columns.swaplevel(1, 2)
        ece_and_recall_cross_ns[correct_order].to_csv(args.model_output_folder+"ECE_and_Metric_cross_ns.csv", sep="\t")

if __name__ == "__main__":
    main()