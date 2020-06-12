from whoosh.index import create_in, open_dir
from whoosh.fields import *
from whoosh import scoring
from whoosh.qparser import QueryParser

import random
import os
import logging
import traceback

class RandomNegativeSampler():

    def __init__(self, candidates, num_candidates_samples, seed=42):
        random.seed(seed)
        self.candidates = candidates
        self.num_candidates_samples = num_candidates_samples
        self.name = "RandomNS"
        
    def sample(self, _, relevant_doc):
        sampled = [d for d in random.sample(self.candidates, self.num_candidates_samples) if d != relevant_doc]
        while len(sampled) != self.num_candidates_samples:
            sampled = [d for d in random.sample(self.candidates, self.num_candidates_samples) if d != relevant_doc]
        return sampled


class TfIdfNegativeSampler():

    def __init__(self, candidates, num_candidates_samples, path_index, seed=42):
        random.seed(seed)
        self.candidates = candidates
        self.num_candidates_samples = num_candidates_samples
        self.path_index  = path_index # self.args.data_folder+self.args.task+"/indexdir_{}".format(self.data_partition)
        self.name = "TfIdfNS"
        self._create_index()

    def _create_index(self):
        schema = Schema(id=ID(stored=True), content=TEXT(stored=True))
        
        if not os.path.isdir(self.path_index):
            os.makedirs(self.path_index)
            self.ix = create_in(self.path_index, schema)
            writer = self.ix.writer()
            for idx, candidate in enumerate(self.candidates):
                writer.add_document(id=u"doc_{}".format(idx),
                                    content=candidate)
            writer.commit()
        else:
            self.ix = open_dir(self.path_index)        

    def sample(self, query_str, relevant_doc):
        # # with self.ix.searcher(weighting=scoring.BM25F()) as searcher:
        with self.ix.searcher(weighting=scoring.TF_IDF()) as searcher:
            try:
                query = QueryParser("content", self.ix.schema).parse(query_str)
                results = searcher.search(query)
                sampled = [r["content"] for r in results[:self.num_candidates_samples] if r["content"] != relevant_doc]
                a = 0/0
            except Exception as e:
                logging.info("Error on query: {}\n\n".format(query_str))
                logging.info(traceback.format_exception(*sys.exc_info()))
                sampled = []

            #If not enough samples are matched, fill with random samples.
            while len(sampled) != self.num_candidates_samples: 
                sampled = sampled + \
                    [d for d in random.sample(self.candidates, self.num_candidates_samples-len(sampled))  
                        if d != relevant_doc]
        # logging.info("query {}".format(query_str))
        # logging.info("sampled {}".format(sampled))
        return sampled