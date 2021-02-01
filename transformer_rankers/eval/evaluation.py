from IPython import embed
import pytrec_eval

METRICS = {'map',
           'recip_rank',
           'ndcg_cut',
           'recall'}

RECALL_AT_W_CAND = {
                    'R_10@1',
                    'R_10@2', 
                    'R_10@5',
                    'R_2@1'
                    }

def recall_at_with_k_candidates(preds, labels, k, at):
    """
    Calculates recall with k candidates. labels list must be sorted by relevance.

    Args:
        preds: float list containing the predictions.
        labels: float list containing the relevance labels.
        k: number of candidates to consider.
        at: threshold to cut the list.
        
    Returns: float containing Recall_k@at
    """
    num_rel = len([l for l in labels if l>=1])
    #'removing' candidates (relevant has to be in first positions in labels)
    preds = preds[:k]
    labels = labels[:k]

    sorted_labels = [x for _,x in sorted(zip(preds, labels), reverse=True)]
    hits = len([l for l in sorted_labels[:at] if l>=1])    
    return hits/num_rel

def evaluate_models(results):
    """
    Calculate METRICS for each model in the results dict
    
    Args:
        results: dict containing one key for each model and inside them pred and label keys. 
        For example:    
             results = {
              'model_1': {
                 'preds': [[1,2],[1,2]],
                 'labels': [[1,2],[1,2]]
               }
            }.
    Returns: dict with the METRIC results per model and query.
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

        for query in qrel.keys(): 
            preds = []
            labels = []
            for doc in run[query].keys():
                preds.append(run[query][doc])
                labels.append(qrel[query][doc])
            
            for recall_metric in RECALL_AT_W_CAND:
                cand = int(recall_metric.split("@")[0].split("R_")[1])
                at = int(recall_metric.split("@")[-1])
                results[model]['eval'][query][recall_metric] = recall_at_with_k_candidates(preds, labels, cand, at)
    return results