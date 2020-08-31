from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from IPython import embed
from scipy import stats
import pandas as pd
import argparse
import logging

import numpy as np
import scipy.stats

np.random.seed(42)

def confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_folder", default=None,
                        type=str, required=True, help="the folder containing predictions and uncertainties.")
    parser.add_argument("--data_path", default=None,
                        type=str, required=True, help="the folder containing clariq data folder.")
    args = parser.parse_args()

    uncertainties = pd.read_csv(args.run_folder+"/uncertainties.csv")
    point_estimate_predictions = pd.read_csv(args.run_folder+"/predictions.csv")
    sthocastic_predictions = pd.read_csv(args.run_folder+"/predictions_with_dropout.csv")
    sthocastic_predictions_softmax = pd.read_csv(args.run_folder+"/predictions_with_dropout_softmax.csv")
    point_estimate_predictions_softmax = pd.read_csv(args.run_folder+"/predictions_softmax.csv")
    #Original data
    train = pd.read_csv(args.data_path+"clariq/train_original.tsv", sep="\t")
    dev = pd.read_csv(args.data_path+"clariq/dev.tsv", sep="\t")
    train = train[["initial_request", "clarification_need"]].drop_duplicates()
    dev = dev[["initial_request", "clarification_need"]].drop_duplicates()
    all_df = pd.concat([train, dev])
    all_df.columns = ["query", "label"]

    #Data used for experiment (we split test into valid and test)
    train = pd.read_csv(args.data_path+"clariq/train.tsv", sep="\t")[["query"]].drop_duplicates()
    dev = pd.read_csv(args.data_path+"clariq/valid.tsv", sep="\t")[["query"]].drop_duplicates()
    both = pd.concat([train, dev]).reset_index(drop=True)
    
    both_with_labels = both.merge(all_df, on=["query"])

    cv_folds = 10
    X = point_estimate_predictions.fillna(0.0).values
    y = both_with_labels["label"].fillna(0.0).values

    scores_random = cross_val_score(DummyClassifier(strategy='uniform', random_state=0), X, y, cv=cv_folds, scoring='f1_macro')
    logging.info("Random predictions.")
    logging.info(scores_random)
    logging.info("F1: %0.2f (+/- %0.2f)\n" % (scores_random.mean(), confidence_interval(scores_random)))

    # clf = RandomForestClassifier(max_depth=8, random_state=0)
    clf = GradientBoostingClassifier(max_depth=8, random_state=0)
    X = point_estimate_predictions.fillna(0.0).values
    scores_baseline = cross_val_score(clf, X[:,0:10], y, cv=cv_folds, scoring='f1_macro')
    logging.info("Using point estimate predictions.")
    logging.info(scores_baseline)
    logging.info("F1: %0.2f (+/- %0.2f)\n" % (scores_baseline.mean(), confidence_interval(scores_baseline)))

    # X = sthocastic_predictions.values
    # scores = cross_val_score(clf, X, y, cv=cv_folds, scoring='f1_macro')
    # logging.info("Using stochastic predictions.")
    # logging.info(scores)
    # logging.info("F1: %0.2f (+/- %0.2f)\n" % (scores.mean(), confidence_interval(scores)))

    X = uncertainties.fillna(0.0).values
    scores = cross_val_score(clf, X[:,0:10], y, cv=cv_folds, scoring='f1_macro')
    logging.info("Using uncertainties.")
    logging.info(scores)
    logging.info("F1: %0.2f (+/- %0.2f)\n" % (scores.mean(), confidence_interval(scores)))

    # embed()
    X = point_estimate_predictions[point_estimate_predictions.columns[0:10]].\
        join(uncertainties[uncertainties.columns[0:10]]).fillna(0.0).values
    scores_proposed = cross_val_score(clf, X, y, cv=cv_folds, scoring='f1_macro')
    logging.info("Using uncertainties and stochastic predictions.")
    logging.info(scores_proposed)
    logging.info("F1: %0.2f (+/- %0.2f)\n" % (scores_proposed.mean(), confidence_interval(scores_proposed)))

    logging.info("Paired t-test for the Preds vs Random.")
    logging.info(stats.ttest_rel(scores_random, scores_baseline))
    logging.info("Paired t-test for the Preds+Uncertainties vs Random.")
    logging.info(stats.ttest_rel(scores_random, scores_proposed))
    logging.info("Paired t-test for the Preds+Uncertainties vs Preds.")
    logging.info(stats.ttest_rel(scores_baseline, scores_proposed))

    # embed()

if __name__ == "__main__":
    main()