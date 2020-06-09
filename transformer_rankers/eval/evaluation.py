import pytrec_eval

METRICS = {'map',
           'recip_rank',
           'ndcg_cut',
           'recall'}

def evaluate_models(results):
    """
    Calculate METRICS for each model in the results dict
    ----
    Input example:
    # results = {
    #  'model_1': {
    #     'preds': [[1,2],[1,2]],
    #     'labels': [[1,2],[1,2]]
    #   }
    #}
    """

    for model in results.keys():
        preds = results[model]['preds']
        labels = results[model]['labels']
        run = {}
        qrel = {}
        for i, p in enumerate(preds):
            run['q{}'.format(i+1)] = {}
            qrel['q{}'.format(i+1)] = {}
            for j, _ in enumerate(range(len(p))):
                run['q{}'.format(i+1)]['d{}'.format(j+1)] = float(preds[i][j])
                qrel['q{}'.format(i + 1)]['d{}'.format(j + 1)] = int(labels[i][j])
        evaluator = pytrec_eval.RelevanceEvaluator(qrel, METRICS)
        results[model]['eval'] = evaluator.evaluate(run)
    return results