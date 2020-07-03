from transformer_rankers.eval import evaluation

def evaluate(preds, labels):
    qrels = {}
    qrels['model'] = {}
    qrels['model']['preds'] = preds
    qrels['model']['labels'] = labels

    results = evaluation.evaluate_models(qrels)
    return results

def evaluate_and_aggregate(preds, labels, metrics):
    """
    Calculate evaluation metrics for a pair of preds and labels.
    
    Aggregates the results only for the evaluation metrics in metrics arg.

    Args:
        preds: list of lists of floats with predictions for each query.
        labels: list of lists with of floats with relevance labels for each query.
        metrics: list of str with the metrics names to aggregate.
        
    Returns: dict with the METRIC results per model and query.
    """
    results = evaluate(preds, labels)

    agg_results = {}
    for metric in metrics:
        res = 0
        per_q_values = []
        for q in results['model']['eval'].keys():
            per_q_values.append(results['model']['eval'][q][metric])
            res += results['model']['eval'][q][metric]
        res /= len(results['model']['eval'].keys())
        agg_results[metric] = res

    return agg_results