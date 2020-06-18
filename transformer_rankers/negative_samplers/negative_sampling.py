from sentence_transformers import SentenceTransformer
from IPython import embed

from whoosh.index import create_in, open_dir
from whoosh.fields import *
from whoosh import scoring
from whoosh.qparser import QueryParser, syntax

import numpy as np
import scipy.spatial
import random
import os
import logging
import traceback
import json
import pickle
import faiss

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
from pyserini.search import SimpleSearcher

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

class TfIdfNegativeSamplerWhoosh():

    def __init__(self, candidates, num_candidates_samples, path_index, seed=42):
        random.seed(seed)
        self.candidates = candidates
        self.num_candidates_samples = num_candidates_samples
        self.path_index  = path_index
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
                query = QueryParser("content", self.ix.schema, group=syntax.OrGroup).parse(query_str)
                results = searcher.search(query)
                sampled = [r["content"] for r in results[:self.num_candidates_samples] if r["content"] != relevant_doc]
            except Exception as e:
                logging.info("Error on query: {}\n\n".format(query_str))
                logging.info(traceback.format_exception(*sys.exc_info()))
                sampled = []
            
            #If not enough samples are matched, fill with random samples.
            while len(sampled) != self.num_candidates_samples: 
                logging.info("Sampling remaining cand for query {}".format(query_str))
                sampled = sampled + \
                    [d for d in random.sample(self.candidates, self.num_candidates_samples-len(sampled))  
                        if d != relevant_doc]
        # logging.info("query {}".format(query_str))
        # logging.info("sampled {}".format(sampled))
        return sampled

class BM25NegativeSamplerPyserini():

    def __init__(self, candidates, num_candidates_samples, path_index, sample_data, anserini_folder, seed=42):
        random.seed(seed)
        self.candidates = candidates
        self.num_candidates_samples = num_candidates_samples
        self.path_index  = path_index
        self.name = "BM25NS"
        self.sample_data = sample_data
        self.anserini_folder = anserini_folder
        self._create_index()

        self.searcher = SimpleSearcher(self.path_index+"anserini_index")
        self.searcher.set_bm25(0.9, 0.4)

    def _generate_anserini_json_collection(self):
        documents = []
        doc_set = set()
        doc_id = 0
        for candidate in self.candidates:
            documents.append({'id': doc_id,
                               'contents': candidate})
            doc_id+=1
        return documents

    def _create_index(self):
        #Create json document files.
        json_files_path = self.path_index+"json_documents_cand_{}".format(self.sample_data)
        if not os.path.isdir(json_files_path):
            os.makedirs(json_files_path)
            docs = self._generate_anserini_json_collection()
            for i, doc in enumerate(docs):
                with open(json_files_path+'/docs{:02d}.json'.format(i), 'w', encoding='utf-8', ) as f:
                    f.write(json.dumps(doc) + '\n')        

            #Run index java command
            os.system("sh {}target/appassembler/bin/IndexCollection -collection JsonCollection"   \
                        " -generator DefaultLuceneDocumentGenerator -threads 9 -input {}" \
                        " -index {}anserini_index -storePositions -storeDocvectors -storeRaw". \
                        format(self.anserini_folder, json_files_path, self.path_index))

    def sample(self, query_str, relevant_doc, max_query_len = 512):        
        #Some long queryies exceeds the maxClauseCount from anserini, so we cut from right to left.
        query_str = query_str[-max_query_len:]        
        sampled = [ hit.raw for hit in self.searcher.search(query_str, k=self.num_candidates_samples) if hit.raw != relevant_doc]        
        while len(sampled) != self.num_candidates_samples: 
                # logging.info("Sampling remaining cand for query {} ...".format(query_str[0:100]))
                sampled = sampled + \
                    [d for d in random.sample(self.candidates, self.num_candidates_samples-len(sampled))  
                        if d != relevant_doc]
        return sampled

class SentenceBERTNegativeSampler():

    def __init__(self, candidates, num_candidates_samples, embeddings_file, sample_data, seed=42):
        random.seed(seed)
        self.candidates = candidates
        self.num_candidates_samples = num_candidates_samples        
        self.name = "SentenceBERTNS"
        self.sample_data = sample_data
        self.embeddings_file = embeddings_file

        self._calculate_sentence_embeddings()
        self._build_faiss_index()

    def _calculate_sentence_embeddings(self, pre_trained_model='bert-base-nli-stsb-mean-tokens'):
        self.model = SentenceTransformer(pre_trained_model)
        embeds_file_path = "{}_n_sample_{}".format(self.embeddings_file, self.sample_data)
        if not os.path.isfile(embeds_file_path):
            logging.info("Calculating embeddings for the candidates.")
            self.candidate_embeddings = self.model.encode(self.candidates)
            with open(embeds_file_path, 'wb') as f:
                pickle.dump(self.candidate_embeddings, f)
        else:
            with open(embeds_file_path, 'rb') as f:
                self.candidate_embeddings = pickle.load(f)
    
    def _build_faiss_index(self):        
        self.index = faiss.IndexFlatL2(self.candidate_embeddings[0].shape[0])   # build the index
        self.index.add(np.array(self.candidate_embeddings))
        logging.info("Faiss index has a total of {} candidates".format(self.index.ntotal))

    def sample(self, query_str, relevant_doc):
        query_embedding = self.model.encode([query_str], show_progress_bar=False)
        
        distances, idxs = self.index.search(np.array(query_embedding), self.num_candidates_samples)        
        sampled = [self.candidates[idx] for idx in idxs[0] if self.candidates[idx]!=relevant_doc]
        
        while len(sampled) != self.num_candidates_samples: 
                # logging.info("Sampling remaining cand for query {} ...".format(query_str[0:100]))
                sampled = sampled + \
                    [d for d in random.sample(self.candidates, self.num_candidates_samples-len(sampled))  
                        if d != relevant_doc]
        return sampled