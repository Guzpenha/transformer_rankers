from sklearn import preprocessing
from IPython import embed
import random
import os
import logging

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
import pyterrier as pt
pt.init()
import pandas as pd
import re

class BM25NegativeSamplerPyterrier():
    """
    Sample candidates from a list of candidates using BM25 from Pyterrier

    Args:
        candidates: list of str containing the candidates
        num_candidates_samples: int containing the number of negative samples for each query.        
        path_index: str containing the path to create/load the  index.
        sample_data: int indicating amount of candidates in the index (-1 if all)            
        set_rm3: boolean indicating whether to use rm3 or not.
        seed: int with the random seed
    """
    def __init__(self, candidates, num_candidates_samples, path_index, sample_data, set_rm3=False, seed=42):
        random.seed(seed)
        self.candidates = candidates
        self.candidates_df = pd.DataFrame(self.candidates, columns=["candidate"]).reset_index().rename(columns={'index': 'docno'})
        self.candidates_df['docno'] = self.candidates_df['docno'].astype(str)
        self.num_candidates_samples = num_candidates_samples
        self.path_index  = path_index
        if set_rm3:
            self.name = "BM25RM3NS_pyterrier"
        else:
            self.name = "BM25NS_pyterrier"
        self.sample_data = sample_data
        self._create_index()

        self.bm25_pipeline = pt.BatchRetrieve(self.indexref, wmodel="BM25") % self.num_candidates_samples

    def _create_index(self):
        """
        Index candidates in case they are not already indexed.
        """
        index_path_with_cand = self.path_index+"_documents_cand_{}".format(self.sample_data)
        if not os.path.isdir(index_path_with_cand):
            os.makedirs(index_path_with_cand, exist_ok=True)
            def from_list_gen():
                for i, cand in enumerate(self.candidates):
                    yield {'docno': i, 'text': cand}
            iter_indexer = pt.IterDictIndexer(index_path_with_cand)
            self.indexref = iter_indexer.index(from_list_gen(), meta=['docno', 'text'])
        self.indexref = pt.IndexRef.of(index_path_with_cand)

    def sample(self, query_str, relevant_docs, max_query_len = 512):
        """
        Samples from a list of candidates using BM25.
        
        If the samples match the relevant doc, 
        then removes it and re-samples randomly.

        Args:
            query_str: the str of the query to be used for BM25
            relevant_docs: list with the str of the relevant documents, to avoid sampling them as negative sample.
            max_query_len: int containing the maximum number of characters to use as input. (Very long queries will raise a maxClauseCount from anserini.)                

        Returns:
            First the sampled_documents, their respective scores and then indicators if the NS retrieved the relevant
            document, and if so at which position.
        """              
        query_str = re.sub('[\W_]', ' ',  query_str)
        query_str = query_str[-max_query_len:]

        sampled_initial_df = self.bm25_pipeline.transform(pd.DataFrame([[0, query_str]], columns=["qid", "query"]))
        sampled_initial = sampled_initial_df.merge(self.candidates_df, on=["docno"])[['candidate','score']].values.tolist()        

        was_relevant_sampled = False
        relevant_doc_rank = -1
        sampled = []
        scores = []
        for i, ds in enumerate(sampled_initial):
            doc, score = ds
            if doc in relevant_docs:
                was_relevant_sampled = True
                relevant_doc_rank = i
            else:
                sampled.append(doc)
                scores.append(score)

        scores_for_random=[0 for i in range(self.num_candidates_samples-len(sampled))]
        while len(sampled) != self.num_candidates_samples: 
                sampled = sampled + \
                    [d for d in random.sample(self.candidates, self.num_candidates_samples-len(sampled))  
                        if d not in relevant_docs]

        if len(scores) != 0: #corner case where only 1 sample and it is the relevant doc.
            normalized_scores = preprocessing.minmax_scale(scores, feature_range=(0.01, 0.99))
        else:
            normalized_scores = []
        
        normalized_scores = list(normalized_scores) + scores_for_random
        return sampled, normalized_scores, was_relevant_sampled, relevant_doc_rank