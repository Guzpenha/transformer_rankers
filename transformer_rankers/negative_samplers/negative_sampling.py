from sentence_transformers import SentenceTransformer
from sklearn import preprocessing
from IPython import embed

import numpy as np
import scipy.spatial
import random
import os
import logging
import traceback
import json
import pickle
import faiss


PYSERINI_USABLE = True
if os.path.isdir("/usr/lib/jvm/java-11-openjdk-amd64"):
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
    from pyserini.search import SimpleSearcher
else:
    PYSERINI_USABLE = False
    logging.info("No java found at /usr/lib/jvm/java-11-openjdk-amd64.")

class RandomNegativeSampler():
    """
    Randomly sample candidates from a list of candidates.

    Args:
        candidates: list of str containing the candidates
        num_candidates_samples: int containing the number of negative samples for each query.
    """
    def __init__(self, candidates, num_candidates_samples, seed=42):
        random.seed(seed)
        self.candidates = candidates
        self.num_candidates_samples = num_candidates_samples
        self.name = "RandomNS"
        
    def sample(self, query_str, relevant_docs):
        """
        Samples from a list of candidates randomly.
        
        If the samples match the relevant doc, 
        then removes it and re-samples.

        Args:
            query_str: the str of the query. Not used here.
            relevant_docs: list with the str of the relevant documents, to avoid sampling them as negative sample.

        Returns:            
             First the sampled_documents, their respective scores and then indicators if the NS retrieved the relevant
             document, and if so at which position.
        """
        sampled_initial = random.sample(self.candidates, self.num_candidates_samples)
        was_relevant_sampled = False
        relevant_doc_rank = -1
        sampled = []
        for i, d in enumerate(sampled_initial):
            if d in relevant_docs:
                was_relevant_sampled = True
                relevant_doc_rank = i
            else:
                sampled.append(d)

        while len(sampled) != self.num_candidates_samples:
            sampled = [d for d in random.sample(self.candidates, self.num_candidates_samples) if d not in relevant_docs]
        return sampled, [random.uniform(0, 0.99) for i in range(len(sampled))], was_relevant_sampled, relevant_doc_rank

if PYSERINI_USABLE:
    class BM25NegativeSamplerPyserini():
        """
        Sample candidates from a list of candidates using BM25.

        The class uses anserini and pyserini which requires JAVA and a installation of anserini.
        It first generates the candidates, saving then to files, then creates the index via
        anserini IndexCollection.

        Args:
            candidates: list of str containing the candidates
            num_candidates_samples: int containing the number of negative samples for each query.        
            path_index: str containing the path to create/load the anserini index.
            sample_data: int indicating amount of candidates in the index (-1 if all)
            anserini_folder: str containing the bin <anserini_folder>/target/appassembler/bin/IndexCollection
            set_rm3: boolean indicating whether to use rm3 or not.
            seed: int with the random seed
        """
        def __init__(self, candidates, num_candidates_samples, path_index, sample_data, anserini_folder, set_rm3=False, seed=42):
            random.seed(seed)
            self.candidates = candidates
            self.num_candidates_samples = num_candidates_samples
            self.path_index  = path_index
            if set_rm3:
                self.name = "BM25RM3NS"
            else:
                self.name = "BM25NS"
            self.sample_data = sample_data
            self.anserini_folder = anserini_folder
            self._create_index()

            self.searcher = SimpleSearcher(self.path_index+"anserini_index")
            self.searcher.set_bm25(0.9, 0.4)
            if set_rm3:
                self.searcher.set_rm3()

        def _generate_anserini_json_collection(self):
            """
            From a list of str documents to the documents in the anserini expected json format.
            """
            documents = []
            doc_set = set()
            doc_id = 0
            for candidate in self.candidates:
                documents.append({'id': doc_id,
                                'contents': candidate})
                doc_id+=1
            return documents

        def _create_index(self):
            """
            Index candidates in case they are not already indexed.
            """
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
            #Some long queryies exceeds the maxClauseCount from anserini, so we cut from right to left.
            query_str = query_str[-max_query_len:]
            sampled_initial = [ (json.loads(hit.raw)['contents'], hit.score) for hit in self.searcher.search(query_str, k=self.num_candidates_samples)]
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

else:
     class BM25NegativeSamplerPyserini():

        def __init__(self, candidates, num_candidates_samples, path_index, sample_data, anserini_folder, set_rm3=False, seed=42):
            self.candidates = candidates
            self.num_candidates_samples = num_candidates_samples
            self.path_index  = path_index
            if set_rm3:
                self.name = "BM25RM3NS"
            else:
                self.name = "BM25NS"
            self.sample_data = sample_data
            self.anserini_folder = anserini_folder
        
        def sample(self, query_str, relevant_doc, max_query_len = 512):
             logging.info("no Java installed, pyserini requires java.")
             return None, None, None

class SentenceBERTNegativeSampler():
    """
    Sample candidates from a list of candidates using dense embeddings from sentenceBERT.

    Args:
        candidates: list of str containing the candidates
        num_candidates_samples: int containing the number of negative samples for each query.
        embeddings_file: str containing the path to cache the embeddings.
        sample_data: int indicating amount of candidates in the index (-1 if all)
        pre_trained_model: str containing the pre-trained sentence embedding model, 
            e.g. bert-base-nli-stsb-mean-tokens.
    """
    def __init__(self, candidates, num_candidates_samples, embeddings_file, sample_data, 
                pre_trained_model='bert-base-nli-stsb-mean-tokens', seed=42):
        random.seed(seed)
        self.candidates = candidates
        self.num_candidates_samples = num_candidates_samples
        self.pre_trained_model = pre_trained_model

        self.model = SentenceTransformer(self.pre_trained_model)
        #extract the name of the folder with the pre-trained sentence embedding        
        if os.path.isdir(self.pre_trained_model):
            self.pre_trained_model = self.pre_trained_model.split("/")[-1] 

        self.name = "SentenceBERTNS_"+self.pre_trained_model
        self.sample_data = sample_data
        self.embeddings_file = embeddings_file

        self._calculate_sentence_embeddings()
        self._build_faiss_index()

    def _calculate_sentence_embeddings(self):
        """
        Calculates sentenceBERT embeddings for all candidates.
        """
        embeds_file_path = "{}_n_sample_{}_pre_trained_model_{}".format(self.embeddings_file,
                                                                        self.sample_data,
                                                                        self.pre_trained_model)
        if not os.path.isfile(embeds_file_path):
            logging.info("Calculating embeddings for the candidates.")
            self.candidate_embeddings = self.model.encode(self.candidates)
            with open(embeds_file_path, 'wb') as f:
                pickle.dump(self.candidate_embeddings, f)
        else:
            with open(embeds_file_path, 'rb') as f:
                self.candidate_embeddings = pickle.load(f)
    
    def _build_faiss_index(self):
        """
        Builds the faiss indexes containing all sentence embeddings of the candidates.
        """
        self.index = faiss.IndexFlatL2(self.candidate_embeddings[0].shape[0])   # build the index
        self.index.add(np.array(self.candidate_embeddings))
        logging.info("There is a total of {} candidates.".format(len(self.candidates)))
        logging.info("There is a total of {} candidate embeddings.".format(len(self.candidate_embeddings)))
        logging.info("Faiss index has a total of {} candidates".format(self.index.ntotal))

    def sample(self, query_str, relevant_docs):
        """
        Samples from a list of candidates using dot product sentenceBERT similarity.
        
        If the samples match the relevant doc, then removes it and re-samples randomly.
        The method uses faiss index to be efficient.

        Args:
            query_str: the str of the query to be used for the dense similarity matching.
            relevant_docs: list with the str of the relevant documents, to avoid sampling them as negative sample.
            
        Returns:
            First the sampled_documents, their respective scores and then indicators if the NS retrieved the relevant
            document, and if so at which position.
        """
        query_embedding = self.model.encode([query_str], show_progress_bar=False)
        
        distances, idxs = self.index.search(np.array(query_embedding), self.num_candidates_samples)
        sampled_initial = [self.candidates[idx] for idx in idxs[0]]
        distances = distances[0]

        was_relevant_sampled = False
        relevant_doc_rank = -1
        sampled = []
        scores = []
        for i, d in enumerate(sampled_initial):
            if d in relevant_docs:
                was_relevant_sampled = True
                relevant_doc_rank = i
            else:
                sampled.append(d)
                scores.append(distances[i])

        scores_for_random=[0 for i in range(self.num_candidates_samples-len(sampled))]
        while len(sampled) != self.num_candidates_samples: 
                sampled = sampled + \
                    [d for d in random.sample(self.candidates, self.num_candidates_samples-len(sampled))  
                        if d not in relevant_docs]
        if len(scores) != 0: #corner case where only 1 sample and it is the relevant doc.
            normalized_scores = preprocessing.minmax_scale(scores, feature_range=(0.01, 0.99))
        else:
            normalized_scores = []
        
        normalized_scores = 1-normalized_scores # similarity instead of distances.
        normalized_scores = list(normalized_scores) + scores_for_random
        return sampled, normalized_scores, was_relevant_sampled, relevant_doc_rank
