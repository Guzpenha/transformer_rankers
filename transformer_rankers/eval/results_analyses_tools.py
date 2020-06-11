from transformer_rankers.eval.evaluation import *

def calculate_effectiveness(predictions_df):
    qrels = {}
    qrels['model'] = {}
    qrels['model']['preds'] = predictions_df.values
    # only first doc is relevant -> [1, 0, 0, ..., 0]
    labels = [[1] + ([0] * (len(predictions_df.columns[1:])))
                for _ in range(predictions_df.shape[0])]
    qrels['model']['labels'] = labels

    return evaluate_models(qrels)