from sentence_transformers import SentenceTransformer
from transformers import pipeline, Conversation, ConversationalPipeline
from scipy.spatial import distance
from sklearn import preprocessing
from IPython import embed
from faiss.contrib.ondisk import merge_ondisk

import numpy as np
import scipy.spatial
import random
import os
import logging
import traceback
import json
import pickle
import faiss
import re
import shutil
import warnings
import math

PYSERINI_USABLE = True
if os.path.isdir("/usr/lib/jvm/java-11-openjdk-amd64"):
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
    from pyserini.search import SimpleSearcher
    from pyserini.dsearch import SimpleDenseSearcher, TctColBertQueryEncoder
    from pyserini.index import IndexReader
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
            query_str: the str of the query. Not used here for random sampling.
            relevant_docs: list with the str of the relevant documents, to avoid sampling them as negative sample.

        Returns:
             sampled_documents: list with the str of the sampled documents            
             scores: list with the size of sampled_documents containing their respective scores
             was_relevant_sampled: boolean indicating if one of the relevant documents would be sampled (we remove the relevant docs from sampled_documents)
             relevant_rank: -1 if was_relevant_sampled=False and the position of the relevant otherwise.
                    This does not work well if there are multiple relevants, i.e. only the last position is returned
             relevant_docs_scores: list of float with the negative sampling model scores for the list of relevant_docs
        """
        #Since random is not a model per se, it makes sense to return 1 for the scores of the relevant docs.
        relevant_docs_scores = [1 for _ in range(len(relevant_docs))]
        sampled_docs_scores = [0 for _ in range(self.num_candidates_samples)]
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
        return sampled, sampled_docs_scores, was_relevant_sampled, relevant_doc_rank, relevant_docs_scores

if PYSERINI_USABLE:
    class DenseRetrievalNegativeSamplerPyserini():
        """
        Sample candidates from a list of candidates using dense searcher from pyserini.

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
        def __init__(self, candidates, num_candidates_samples, path_index, sample_data, anserini_folder, seed=42):
            random.seed(seed)
            self.candidates = candidates
            self.num_candidates_samples = num_candidates_samples
            self.path_index  = path_index
            self.name = "TCT-ColBERT"
            self.sample_data = sample_data
            self.anserini_folder = anserini_folder
            self._create_index()

            encoder = TctColBertQueryEncoder('castorini/tct_colbert-msmarco')
            self.searcher = SimpleDenseSearcher(self.path_index+"anserini_index_dense", encoder)
            self.doc_searcher = SimpleSearcher(self.path_index+"anserini_index")

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
            
            if not os.path.isdir(self.path_index+'anserini_index_dense'):
                os.system("python -m pyserini.dindex --corpus {}" \
                            " --encoder \'castorini/tct_colbert-msmarco\'" \
                            " --index {}anserini_index_dense/" \
                            " --batch 64" \
                            " --device cuda:0".format(json_files_path, self.path_index))

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
             sampled_documents: list with the str of the sampled documents            
             scores: list with the size of sampled_documents containing their respective scores
             was_relevant_sampled: boolean indicating if one of the relevant documents would be sampled (we remove the relevant docs from sampled_documents)
             relevant_rank: -1 if was_relevant_sampled=False and the position of the relevant otherwise.
                    This does not work well if there are multiple relevants, i.e. only the last position is returned
             relevant_docs_scores: list of float with the negative sampling model scores for the list of relevant_docs
            """
            # TODO how to get scores from SimpleDenseSearcher for a combination of query and doc 
            relevant_docs_scores = [1 for _ in range(len(relevant_docs))]
            #Some long queryies exceeds the maxClauseCount from anserini, so we cut from right to left.
            query_str = query_str[-max_query_len:]
            sampled_initial = [ (json.loads(self.doc_searcher.doc(hit.docid).raw())['contents'], hit.score) for hit in self.searcher.search(query_str, k=self.num_candidates_samples)]
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
            return sampled, normalized_scores, was_relevant_sampled, relevant_doc_rank, relevant_docs_scores

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
        def __init__(self, candidates, num_candidates_samples, path_index, sample_data, anserini_folder, set_rm3=False, seed=42,
            num_expansion_terms=10, num_expansion_docs=10, original_query_weight=0.5):
            random.seed(seed)
            self.score_relevant_docs = False
            self.candidates = candidates
            self.num_candidates_samples = num_candidates_samples
            self.path_index  = path_index
            self.set_rm3=set_rm3
            if self.set_rm3:
                self.name = "BM25RM3NS"
            else:
                self.name = "BM25NS"
            self.sample_data = sample_data
            self.anserini_folder = anserini_folder
            self._create_index()

            self.searcher = SimpleSearcher(self.path_index+"anserini_index")
            self.searcher.set_bm25(0.9, 0.4)
            if self.set_rm3:
                self.searcher.set_rm3(fb_terms=num_expansion_terms, fb_docs=num_expansion_docs, original_query_weight=original_query_weight)

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

        def sample(self, query_str, relevant_docs, max_query_len = 512, normalize_scores = False, rel_doc_id = -1):
            """
            Samples from a list of candidates using BM25.
            
            If the samples match the relevant doc, 
            then removes it and re-samples randomly.

            Args:
                query_str: the str of the query to be used for BM25
                relevant_docs: list with the str of the relevant documents, to avoid sampling them as negative sample.
                max_query_len: int containing the maximum number of characters to use as input. (Very long queries will raise a maxClauseCount from anserini.)                

            Returns:
             sampled_documents: list with the str of the sampled documents            
             scores: list with the size of sampled_documents containing their respective scores
             was_relevant_sampled: boolean indicating if one of the relevant documents would be sampled (we remove the relevant docs from sampled_documents)
             relevant_rank: -1 if was_relevant_sampled=False and the position of the relevant otherwise.
                    This does not work well if there are multiple relevants, i.e. only the last position is returned
             relevant_docs_scores: list of float with the negative sampling model scores for the list of relevant_docs
            """
            
            if self.score_relevant_docs:
                query_str = query_str[-max_query_len:]
                index_reader = IndexReader(self.path_index+"anserini_index")
                rel_score = index_reader.compute_query_document_score(rel_doc_id, query_str)
                relevant_docs_scores = [rel_score]
            else:
                relevant_docs_scores = [1 for _ in range(len(relevant_docs))]

            #Some long queryies exceeds the maxClauseCount from anserini, so we cut from right to left.
            query_str = query_str[-max_query_len:]
            sampled_initial = [(json.loads(hit.raw)['contents'], hit.score) for hit in self.searcher.search(query_str, k=self.num_candidates_samples)]
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
                if normalize_scores:
                    scores = preprocessing.minmax_scale(scores, feature_range=(0.01, 0.99))
            else:
                scores = []
            # print(relevant_docs_scores)
            # print(was_relevant_sampled)
            # print(scores)
            scores = list(scores) + scores_for_random
            return sampled, scores, was_relevant_sampled, relevant_doc_rank, relevant_docs_scores

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
                pre_trained_model="all-MiniLM-L6-v2", seed=42, use_cache_for_embeddings=False,
                large_index=False):
        random.seed(seed)
        warnings.filterwarnings("ignore")
        self.use_cache_for_embeddings = use_cache_for_embeddings
        self.score_relevant_docs = False
        self.candidates = candidates
        self.num_candidates_samples = num_candidates_samples
        self.pre_trained_model = pre_trained_model
        self.large_index = large_index

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
        if not self.large_index: # large index will not fit in mem
            embeds_file_path = "{}_n_sample_{}_pre_trained_model_{}".format(self.embeddings_file,
                                                                            self.sample_data,
                                                                            self.pre_trained_model)

            if not os.path.isfile(embeds_file_path) or not self.use_cache_for_embeddings:
                logging.info("Calculating embeddings for the candidates.")
                self.candidate_embeddings = self.model.encode(self.candidates)
                with open(embeds_file_path, 'wb') as f:
                    pickle.dump(self.candidate_embeddings, f)
            else:
                logging.info("Using cached embeddings for candidates.")
                with open(embeds_file_path, 'rb') as f:
                    self.candidate_embeddings = pickle.load(f)
    
    def _build_faiss_index(self):
        """
        Builds the faiss indexes containing all sentence embeddings of the candidates.
        """
        logging.info("Building faiss index")
        if self.large_index:
            # self.index = faiss.index_factory(dim, "OPQ16_64,IVF65536_HNSW32,PQ16")
            bloc_size = 1000000
            embeddings_sample = self.model.encode(random.choices(self.candidates, k=bloc_size))
            dim = embeddings_sample[0].shape[0]
            self.index = faiss.index_factory(dim, "IVF4096,Flat")

            logging.info("Training index")
            ngpus = faiss.get_num_gpus()
            logging.info("Using {} gpus".format(ngpus))
            index_ivf = faiss.extract_index_ivf(self.index)
            clustering_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(index_ivf.d))
            index_ivf.clustering_index = clustering_index
            self.index.train(embeddings_sample)
            index_path = "{}_pre_trained_model_{}_trained.index".format(self.embeddings_file,
                                                                        self.pre_trained_model)
            faiss.write_index(self.index, index_path)

            logging.info("Creating index blocks")
            
            block_num = 0
            for i in range(0, len(self.candidates), bloc_size):
                block_candidates_embeded = self.model.encode(self.candidates[i:i+bloc_size])
                self.index = faiss.read_index(index_path)
                self.index.add(np.array(block_candidates_embeded))
                logging.info("Writing block {}".format(block_num))
                index_path_block = "{}_pre_trained_model_{}_trained_block_{}.index".format(self.embeddings_file,
                    self.pre_trained_model,
                    block_num)
                faiss.write_index(self.index, index_path_block)
                block_num+=1
            
            logging.info("Merging indexes on disk")
            self.index = faiss.read_index(index_path)
            block_f_names = ["{}_pre_trained_model_{}_trained_block_{}.index".format(self.embeddings_file,
                            self.pre_trained_model, i) for i in range(math.ceil(len(self.candidates)/bloc_size))]
            merge_ondisk(self.index, block_f_names, 
                            "{}_pre_trained_model_{}_merged_index.ivfdata".format(self.embeddings_file,
                                                                                 self.pre_trained_model))
            logging.info("There is a total of {} candidates.".format(len(self.candidates)))
            logging.info("Faiss index has a total of {} candidates".format(self.index.ntotal))

        else:
            dim = self.candidate_embeddings[0].shape[0]
            self.index = faiss.IndexFlatIP(dim)   # build the index
            self.index.add(np.array(self.candidate_embeddings))
            logging.info("There is a total of {} candidates.".format(len(self.candidates)))
            logging.info("There is a total of {} candidate embeddings.".format(len(self.candidate_embeddings)))
            logging.info("Faiss index has a total of {} candidates".format(self.index.ntotal))

    def sample(self, query_str, relevant_docs, normalize_scores=False):
        """
        Samples from a list of candidates using dot product sentenceBERT similarity.
        
        If the samples match the relevant doc, then removes it and re-samples randomly.
        The method uses faiss index to be efficient.

        Args:
            query_str: the str of the query to be used for the dense similarity matching.
            relevant_docs: list with the str of the relevant documents, to avoid sampling them as negative sample.
            
        Returns:
            sampled_documents: list with the str of the sampled documents            
            scores: list with the size of sampled_documents containing their respective scores
            was_relevant_sampled: boolean indicating if one of the relevant documents would be sampled (we remove the relevant docs from sampled_documents)
            relevant_rank: -1 if was_relevant_sampled=False and the position of the relevant otherwise.
                    This does not work well if there are multiple relevants, i.e. only the last position is returned
            relevant_docs_scores: list of float with the negative sampling model scores for the list of relevant_docs
        """
        query_embedding = self.model.encode([query_str], show_progress_bar=False)

        if self.score_relevant_docs:
            relevant_docs_embeddings = self.model.encode(relevant_docs, show_progress_bar=False)
            relevant_docs_scores = np.dot(relevant_docs_embeddings, query_embedding[0])
        else:
            relevant_docs_scores = [1 for _ in range(len(relevant_docs))]
        
        similarities, idxs = self.index.search(np.array(query_embedding), self.num_candidates_samples)
        sampled_initial = [self.candidates[idx] for idx in idxs[0]]
        if len(similarities) > 0:
            similarities = similarities[0]
        else:
            # print(query_str)
            # print(sampled_initial)
            similarities = []

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
                scores.append(similarities[i])

        scores_for_random=[0 for i in range(self.num_candidates_samples-len(sampled))]
        while len(sampled) != self.num_candidates_samples: 
                sampled = sampled + \
                    [d for d in random.sample(self.candidates, self.num_candidates_samples-len(sampled))  
                        if d not in relevant_docs]
        if len(scores) != 0: 
            if normalize_scores:
                scores = preprocessing.minmax_scale(scores, feature_range=(0.01, 0.99))
        else: #corner case where only 1 sample and it is the relevant doc.
            scores = []
        scores = list(scores) + scores_for_random
        return sampled, scores, was_relevant_sampled, relevant_doc_rank, relevant_docs_scores

class GenerativeNegativeSamplerForDialogue():
    """
    Generates negative samples on the go based on a generative pre-trained LM.

    As it stands we use it with a conversational pipeline, but this could be expanded to use other pipelines.

    Args:        
        num_candidates_samples: int containing the number of negative samples for each query. 
        pre_trained_model: str containing the pre-trained language model, 
            e.g. bert-base-nli-stsb-mean-tokens.
    """
    def __init__(self, num_candidates_samples, pre_trained_model, seed=42):
        self.num_candidates_samples = num_candidates_samples
        self.pre_trained_model = pre_trained_model

        self.name = "GeneratedCandidateLM_{}".format(self.pre_trained_model)

        self.pipeline = pipeline("conversational", model=pre_trained_model, device=0)
        # embed()

    def sample(self, query_str, relevant_docs, normalize_scores=False, split_context_tokens=["[UTTERANCE_SEP]", "[TURN_SEP]"]):
        """
        Generates negative candidates on the fly using a pre-trained LM.

        Args:
            query_str: the str of the query to be used as input to the LM.
            relevant_docs: list with the str of the relevant documents, not used for this class.
            
        Returns:
            sampled_documents: list with the str of the generated documents            
            scores: list with the size of sampled_documents containing their respective scores
            was_relevant_sampled: boolean indicating if one of the relevant documents would be sampled (we remove the relevant docs from sampled_documents)
            relevant_rank: -1 if was_relevant_sampled=False and the position of the relevant otherwise.
                    This does not work well if there are multiple relevants, i.e. only the last position is returned
            relevant_docs_scores: list of float with the negative sampling model scores for the list of relevant_docs
        """
        # the generative model is not yet scoring the relavant in this code so always 1 
        relevant_docs_scores = [1 for _ in range(len(relevant_docs))]

        for t in split_context_tokens:
            query_str = query_str.replace(t, "[SPLIT]")
        utterances = query_str.split("[SPLIT]")
        utterances = [u.strip() for u in utterances if u.strip()!='']
        # if len(utterances)>1:
        #     past_user_inputs = []
        #     responses = []
        #     for i in range(len(utterances)):
        #         if (i+1) % 2 == 0:
        #             responses.append(utterances[i])
        #         else:
        #             past_user_inputs.append(utterances[i])
        #     conversation = Conversation(past_user_inputs=past_user_inputs,generated_responses=responses)
        #     conversation.add_user_input(utterances[-1])
        # else:
        #     conversation = Conversation(utterances[0])
        
        conversation = Conversation(" ".join(utterances))
        # print(query_str)
        self.pipeline(conversation)
        generated_response = conversation.generated_responses[-1]
        # print(generated_response)
        return [generated_response], [0], False, -1, relevant_docs_scores
        # return sampled, scores, was_relevant_sampled, relevant_doc_rank, relevant_docs_scores