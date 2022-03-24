from dataclasses import dataclass, field
from typing import Dict, List, Optional

from transformers import DataCollator
from transformers.data.processors.utils import InputFeatures
from transformers.data.data_collator import default_data_collator

from IPython import embed
from tqdm import tqdm
from abc import *

import torch.utils.data as data
import numpy as np

import dataclasses
import functools
import operator
import torch
import logging
import random
import os
import pickle

def collate_T2T_batch(batch: List) -> Dict[str, torch.Tensor]:
    """
    Data collator for generation tasks.
    """
    input_ids = torch.stack([torch.tensor(example['input_ids'], dtype=torch.long) for example in batch])
    lm_labels = torch.stack([torch.tensor(example['target_ids'], dtype=torch.long) for example in batch])
    lm_labels[lm_labels[:, :] == 0] = -100
    attention_mask = torch.stack([torch.tensor(example['attention_mask'], dtype=torch.long) for example in batch])
    decoder_attention_mask = torch.stack([torch.tensor(example['target_attention_mask'], dtype=torch.long) for example in batch])

    return {
        'input_ids': input_ids, 
        'attention_mask': attention_mask,
        'labels': lm_labels, 
        'decoder_attention_mask': decoder_attention_mask
    }

class AbstractDataloader(metaclass=ABCMeta):
    """
        Abstract class for the DataLoaders. The class expects only relevant query-doc combinations in the dfs.

        Args:
            train_df: train pandas DataFrame containing columns the first containing the 'query' the second one relevant 'document'.
            val_df: validation pandas DataFrame containing columns the first containing the 'query' the second one relevant 'document'.
            test_df: test pandas DataFrame containing columns the first containing the 'query' the second one relevant 'document'.
            tokenizer: transformer tokenizer.
            negative_sampler_train: negative sampling class for the training set. Has .sample() function.
            negative_sampler_val: negative sampling class for the validation/test set. Has .sample() function.
            train_batch_size: int containing the number of instances in a batch for training.
            val_batch_size: int containing the number of instances in a batch for validation/test.
            max_seq_len: int containing the maximum sentence length when processing inputs.
            sample_data: int containing whether the data was sampled (num_samples) or not (-1).
            cache_path: str with the path to cache the dataset already in torch tensors format.
    """
    def __init__(self, train_df, val_df, test_df, tokenizer,
                 negative_sampler_train, negative_sampler_val, task_type,
                 train_batch_size, val_batch_size, max_seq_len, sample_data,
                 cache_path):

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.negative_sampler_train = negative_sampler_train
        self.negative_sampler_val = negative_sampler_val
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.max_seq_len = max_seq_len
        self.sample_data = sample_data
        self.cache_path = cache_path

        self.num_gpu = torch.cuda.device_count()
        self.task_type = task_type

        self.actual_train_batch_size = self.train_batch_size \
                                       * max(1, self.num_gpu)
        logging.info("Train instances per batch {}".
                     format(self.actual_train_batch_size))

    @abstractmethod
    def get_pytorch_dataloaders(self):
        pass

class QueryDocumentDataLoader(AbstractDataloader):
    def __init__(self, train_df, val_df, test_df, tokenizer,
                 negative_sampler_train, negative_sampler_val, task_type,
                 train_batch_size, val_batch_size, max_seq_len, sample_data,
                 cache_path):
        super().__init__(train_df, val_df, test_df, tokenizer,
                 negative_sampler_train, negative_sampler_val, task_type,
                 train_batch_size, val_batch_size, max_seq_len, sample_data,
                 cache_path)

        if self.task_type == "classification":
            self.data_collator = default_data_collator
        elif self.task_type == "generation":
            special_tokens_dict = {
                'additional_special_tokens': ['not_relevant']
            }
            self.tokenizer.add_special_tokens(special_tokens_dict)
            self.data_collator = collate_T2T_batch

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader

    def _get_train_loader(self):
        dataset = QueryDocumentDataset(self.train_df, self.tokenizer,'train',
                             self.negative_sampler_train, self.task_type,
                             self.max_seq_len, self.sample_data, self.cache_path)
        dataloader = data.DataLoader(dataset,
                                     batch_size=self.actual_train_batch_size,
                                     shuffle=True,
                                     collate_fn=self.data_collator)
        return dataloader

    def _get_val_loader(self):
        dataset = QueryDocumentDataset(self.val_df, self.tokenizer, 'val', 
                            self.negative_sampler_val, self.task_type,
                             self.max_seq_len, self.sample_data, self.cache_path)
        dataloader = data.DataLoader(dataset,
                                     batch_size=self.val_batch_size,
                                     shuffle=False,
                                     collate_fn=self.data_collator)
        return dataloader

    def _get_test_loader(self):
        dataset = QueryDocumentDataset(self.test_df, self.tokenizer, 'test', 
                             self.negative_sampler_val, self.task_type,
                             self.max_seq_len, self.sample_data, self.cache_path)
        dataloader = data.DataLoader(dataset,
                                     batch_size=self.val_batch_size,
                                     shuffle=False,
                                     collate_fn=self.data_collator)
        return dataloader

class QueryDocumentDataset(data.Dataset):
    """
    Dataset for pointwise learning with <Query,Document> pairs.
    Generative transformers are not supported for the cached_mode 'memmap'
    """
    def __init__(self, data, tokenizer, data_partition, 
                negative_sampler, task_type, max_seq_len, sample_data,
                cache_path, cache_mode = "memmap"):
        random.seed(42)

        self.data = data
        self.tokenizer = tokenizer
        self.data_partition = data_partition
        self.negative_sampler = negative_sampler
        self.instances = []
        self.task_type = task_type
        self.max_seq_len = max_seq_len
        self.sample_data = sample_data
        self.cache_path = cache_path
        self.cache_mode = cache_mode

        self._group_relevant_documents()
        if self.cache_mode =='memmap':
            self._cache_instances_memmap()
        elif self.cache_mode =='pickle':
            self._cache_instances_pickle()
        del(self.data)

    def _group_relevant_documents(self):
        """
        Since some datasets have multiple relevants per query, we group them to make NS easier.
        """
        query_col = self.data.columns[0]
        num_rel_documents = self.data.shape[0] #datasets have a format of one line per query and doc combination
        self.data = self.data.groupby(query_col).agg(list).reset_index()
        num_queries = self.data.shape[0]
        self.number_instances = (num_queries * self.negative_sampler.num_candidates_samples) + num_rel_documents

    def _cache_instances_pickle(self):
        """
        WARNING: This is no longer being used as a cache method and memmap should be prefered
        Loads tensors into memory or creates the dataset when it does not exist already.        
        """        
        signature = "pointwise_set_{}_n_cand_docs_{}_ns_sampler_{}_seq_max_l_{}_sample_{}_for_{}_using_{}.pk".\
            format(self.data_partition,
                   self.negative_sampler.num_candidates_samples,
                   self.negative_sampler.name,
                   self.max_seq_len,
                   self.sample_data,
                   self.task_type,
                   self.tokenizer.__class__.__name__)
        path = self.cache_path + "/" + signature

        if os.path.exists(path):
            with open(path, 'rb') as f:
                logging.info("Loading instances from {}".format(path))
                self.instances = pickle.load(f)
        else:
            logging.info("Generating instances with signature {}".format(signature))

            #Creating labels (currently there is support only for binary relevance)
            if self.task_type == "classification":
                relevant_label = 1
                not_relevant_label = 0
            elif self.task_type == "generation":
                relevant_label = "relevant </s>"
                not_relevant_label = "not_relevant  </s>"
            labels = []
            for r in self.data.itertuples(index=False):
                labels+=([relevant_label] * len(r[1])) #relevant documents are grouped at the second column.
                labels+=([not_relevant_label] * (self.negative_sampler.num_candidates_samples)) # each query has N negative samples.

            examples = []
            for idx, row in enumerate(tqdm(self.data.itertuples(index=False), total=len(self.data))):
                query = row[0]
                relevant_documents = row[1]
                for relevant_document in relevant_documents:
                    examples.append((query, relevant_document))
                ns_candidates, _ , _ , _, _ = self.negative_sampler.sample(query, relevant_documents)
                for ns in ns_candidates:
                    examples.append((query, ns))

            logging.info("Encoding examples using tokenizer.batch_encode_plus().")
            batch_encoding = self.tokenizer(examples, max_length=self.max_seq_len,
                                                      padding="max_length", truncation=True)

            if self.task_type == "generation": 
                target_encodings = self.tokenizer(labels,
                    max_length=10, padding="max_length", truncation=True)
                target_encodings = {
                        "target_ids": target_encodings["input_ids"],
                        "target_attention_mask": target_encodings["attention_mask"]
                    }

            logging.info("Transforming examples to instances format.")
            self.instances = []
            for i in range(len(examples)):
                inputs = {k: batch_encoding[k][i] for k in batch_encoding}
                if self.task_type == "generation":
                    targets = {k: target_encodings[k][i] for k in target_encodings}
                    feature = {**inputs, **targets}
                elif self.task_type == "classification":
                    feature = InputFeatures(**inputs, label=labels[i])
                self.instances.append(feature)
            for idx in range(3):
                logging.info("Set {} Instance {} query \n\n{}[...]\n".format(self.data_partition, idx, examples[idx][0][0:200]))
                logging.info("Set {} Instance {} document \n\n{}\n".format(self.data_partition, idx, examples[idx][1][0:200]))
                logging.info("Set {} Instance {} features \n\n{}\n".format(self.data_partition, idx, self.instances[idx]))
            with open(path, 'wb') as f:
                pickle.dump(self.instances, f)

        logging.info("Total of {} instances were cached.".format(len(self.instances)))

    def _cache_instances_memmap(self):
        """
        Loads the memset numpy matrices in case they exist otherwise create them.
        """
        signature = "pointwise_set_{}_n_cand_docs_{}_ns_sampler_{}_seq_max_l_{}_sample_{}_for_{}_using_{}.dat".\
            format(self.data_partition,
                   self.negative_sampler.num_candidates_samples,
                   self.negative_sampler.name,
                   self.max_seq_len,
                   self.sample_data,
                   self.task_type,
                   self.tokenizer.__class__.__name__)
        path_input_ids = self.cache_path + "/input_ids_" + signature
        if not os.path.exists(path_input_ids):
            logging.info("Generating instances with signature {}".format(signature))
            labels = []
            for r in self.data.itertuples(index=False):
                #relevant documents are grouped at the second column.
                if len(r) > 2: #multiple levels of relevance
                    labels+=r[2]
                else:
                    labels+=([1] * len(r[1]))
                labels+=([0] * (self.negative_sampler.num_candidates_samples)) # each query has N negative samples.

            labels = np.array(labels)
            path_labels = self.cache_path + "/labels_" + signature
            self.mem_maps = {}
            self.mem_maps['labels'] = np.memmap(path_labels, dtype='int', mode='w+', shape=labels.shape)
            self.mem_maps['labels'][:] = labels[:]

            logging.info("Finding negative samples and applying tokenizer.batch_encode_plus().")
            len_each_batch = min(1000, self.data.shape[0])
            number_batches = self.data.shape[0]//len_each_batch
            logging.info("Batches of size {}.".format(len_each_batch))
            for k in ['input_ids', 'token_type_ids', 'attention_mask']:
                path_input = self.cache_path + "/{}_".format(k) + signature
                self.mem_maps[k] = np.memmap(path_input, dtype='int', mode='w+', shape=(labels.shape[0], self.max_seq_len))

            instances_idx=0
            for idx, chunk in enumerate(tqdm(np.array_split(self.data, number_batches))):
                examples = []
                for row in chunk.itertuples(index=False):
                    query = row[0]
                    relevant_documents = row[1]
                    for relevant_document in relevant_documents:
                        examples.append((query, relevant_document))
                    ns_candidates, _ , _ , _, _= self.negative_sampler.sample(query, relevant_documents)
                    for ns in ns_candidates:
                        examples.append((query, ns))

                batch_encoding = self.tokenizer(examples, max_length=self.max_seq_len,
                                                        padding="max_length", truncation=True)
                for k in batch_encoding:
                    np_batch_encoding = np.array(batch_encoding[k])
                    self.mem_maps[k][instances_idx:instances_idx+np_batch_encoding.shape[0]] = np_batch_encoding[:]

                instances_idx+=np_batch_encoding.shape[0]

            del(self.mem_maps)

        self.mem_maps = {}
        for k in ['input_ids', 'token_type_ids', 'attention_mask', 'labels']:
            filename = self.cache_path + "/{}_".format(k) + signature
            logging.info("Loading {} from {}".format(k, filename))
            data_shape = (self.number_instances, self.max_seq_len)
            if k == 'labels':
                data_shape = (self.number_instances)
            self.mem_maps[k] = np.memmap(filename, dtype='int', mode='r', shape=data_shape)        

    def __len__(self):
        return self.number_instances

    def __getitem__(self, index):
        if self.cache_mode == 'pickle':
            return self.instances[index]
        elif self.cache_mode == 'memmap':
            instance_data = {}
            for k in self.mem_maps:
                if k != "labels":
                    instance_data[k] = list(self.mem_maps[k][index])
            inputs = {k: instance_data[k] for k in instance_data}
            feature = InputFeatures(**inputs, label=int(self.mem_maps["labels"][index]))
            return feature

class QueryPosDocNegDocDataLoader(AbstractDataloader):
    def __init__(self, train_df, val_df, test_df, tokenizer,
                 negative_sampler_train, negative_sampler_val, task_type,
                 train_batch_size, val_batch_size, max_seq_len, sample_data,
                 cache_path):
        super().__init__(train_df, val_df, test_df, tokenizer,
                 negative_sampler_train, negative_sampler_val, task_type,
                 train_batch_size, val_batch_size, max_seq_len, sample_data,
                 cache_path)

        if self.task_type == "classification":
            self.data_collator = default_data_collator
        elif self.task_type == "generation":
            special_tokens_dict = {
                'additional_special_tokens': ['not_relevant']
            }
            self.tokenizer.add_special_tokens(special_tokens_dict)
            self.data_collator = collate_T2T_batch

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader

    def _get_train_loader(self):
        dataset = QueryPosDocNegDocDataset(self.train_df, self.tokenizer,'train',
                             self.negative_sampler_train, self.task_type,
                             self.max_seq_len, self.sample_data, self.cache_path)
        dataloader = data.DataLoader(dataset,
                                     batch_size=self.actual_train_batch_size,
                                     shuffle=True,
                                     collate_fn=self.data_collator)
        return dataloader

    def _get_val_loader(self):
        dataset = QueryPosDocNegDocDataset(self.val_df, self.tokenizer, 'val', 
                            self.negative_sampler_val, self.task_type,
                             self.max_seq_len, self.sample_data, self.cache_path)
        dataloader = data.DataLoader(dataset,
                                     batch_size=self.val_batch_size,
                                     shuffle=False,
                                     collate_fn=self.data_collator)
        return dataloader

    def _get_test_loader(self):
        dataset = QueryPosDocNegDocDataset(self.test_df, self.tokenizer, 'test', 
                             self.negative_sampler_val, self.task_type,
                             self.max_seq_len, self.sample_data, self.cache_path)
        dataloader = data.DataLoader(dataset,
                                     batch_size=self.val_batch_size,
                                     shuffle=False,
                                     collate_fn=self.data_collator)
        return dataloader

class QueryPosDocNegDocDataset(data.Dataset):
    """
    Dataset for pairwise learning with <Query,PosDoc,NegDoc> triplets. 
    For each batch both the positive and the negative documents are kept in memory.
    """
    def __init__(self, data, tokenizer, data_partition, 
                negative_sampler, task_type, max_seq_len, sample_data,
                cache_path, generate_inverted_pairs=False):
        random.seed(42)

        self.data = data
        self.tokenizer = tokenizer
        self.data_partition = data_partition
        self.negative_sampler = negative_sampler
        self.instances = []
        self.task_type = task_type
        self.max_seq_len = max_seq_len
        self.sample_data = sample_data
        self.cache_path = cache_path
        self.generate_inverted_pairs = generate_inverted_pairs

        self._group_relevant_documents()
        self._cache_instances()

    def _group_relevant_documents(self):
        """
        Since some datasets have multiple relevants per query, we group them to make NS easier.
        """
        query_col = self.data.columns[0]
        self.data = self.data.groupby(query_col).agg(list).reset_index()

    def _cache_instances(self):
        """
        Loads tensors into memory or creates the dataset when it does not exist already.
        """        
        signature = "pairwise_set_{}_n_cand_docs_{}_ns_sampler_{}_seq_max_l_{}_sample_{}_for_{}_using_{}_inverted_pairs_{}".\
            format(self.data_partition,
                   self.negative_sampler.num_candidates_samples,
                   self.negative_sampler.name,
                   self.max_seq_len,
                   self.sample_data,
                   self.task_type,
                   self.tokenizer.__class__.__name__,
                   self.generate_inverted_pairs)
        path = self.cache_path + "/" + signature

        if os.path.exists(path):
            with open(path, 'rb') as f:
                logging.info("Loading instances from {}".format(path))
                self.instances = pickle.load(f)
        else:            
            logging.info("Generating instances with signature {}".format(signature))

            examples = []
            labels = []
            for idx, row in enumerate(tqdm(self.data.itertuples(index=False), total=len(self.data))):
                query = row[0]
                relevant_documents = row[1]
                ns_candidates, _ ,  _ , _, _= self.negative_sampler.sample(query, relevant_documents)                
                for rel_doc in relevant_documents:
                    if self.data_partition == 'train':
                        for ns_doc in ns_candidates:
                            examples.append((query, rel_doc, ns_doc))
                            labels.append(1)
                            if self.generate_inverted_pairs:
                                examples.append((query, ns_doc, rel_doc))
                                labels.append(0)
                    else:
                        # at prediction time we do not use the second document
                        examples.append((query, rel_doc, rel_doc)) 
                        labels.append(1)
                        for ns_doc in ns_candidates:
                            examples.append((query, ns_doc, ns_doc)) 
                            labels.append(0)

            logging.info("Encoding examples using tokenizer.batch_encode_plus().")
            batch_encoding_pos = self.tokenizer([(e[0], e[1]) for e in examples], 
                                    max_length=self.max_seq_len,padding="max_length", truncation=True)
            batch_encoding_neg = self.tokenizer([(e[0], e[2]) for e in examples], 
                                    max_length=self.max_seq_len,padding="max_length", truncation=True)
            logging.info("Transforming examples to instances format.")
            self.instances = []
            for i in range(len(examples)):
                inputs = {'input_ids_pos': batch_encoding_pos['input_ids'][i],
                          'token_type_ids_pos': batch_encoding_pos['token_type_ids'][i],
                          'attention_mask_pos': batch_encoding_pos['attention_mask'][i],
                          'input_ids_neg': batch_encoding_neg['input_ids'][i],
                          'token_type_ids_neg': batch_encoding_neg['token_type_ids'][i],
                          'attention_mask_neg': batch_encoding_neg['attention_mask'][i],
                          'labels': labels[i]}

                self.instances.append(inputs)

            for idx in range(3):
                logging.info("Set {} Instance {} query \n\n{}[...]\n".format(self.data_partition, idx, examples[idx][0][0:200]))
                logging.info("Set {} Instance {} rel document \n\n{}\n".format(self.data_partition, idx, examples[idx][1][0:200]))
                logging.info("Set {} Instance {} ns document \n\n{}\n".format(self.data_partition, idx, examples[idx][2][0:200]))
                # logging.info("Set {} Instance {} features \n\n{}\n".format(self.data_partition, idx, self.instances[idx]))
            with open(path, 'wb') as f:
                pickle.dump(self.instances, f)

        logging.info("Total of {} instances were cached.".format(len(self.instances)))

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        return self.instances[index]

class WeaklySupervisedQueryDocumentDataLoader(AbstractDataloader):
    def __init__(self, train_df, val_df, test_df, tokenizer,
                 negative_sampler_train, negative_sampler_val, task_type,
                 train_batch_size, val_batch_size, max_seq_len, sample_data,
                 cache_path):
        super().__init__(train_df, val_df, test_df, tokenizer,
                 negative_sampler_train, negative_sampler_val, task_type,
                 train_batch_size, val_batch_size, max_seq_len, sample_data,
                 cache_path)
    
        self.data_collator = default_data_collator

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader

    def _get_train_loader(self):
        dataset = WeaklySupervisedQueryDocumentDataset(self.train_df, self.tokenizer,'train',
                             self.negative_sampler_train, self.task_type,
                             self.max_seq_len, self.sample_data, self.cache_path)
        dataloader = data.DataLoader(dataset,
                                     batch_size=self.actual_train_batch_size,
                                     shuffle=True,
                                     collate_fn=self.data_collator)
        return dataloader

    def _get_val_loader(self):
        dataset = WeaklySupervisedQueryDocumentDataset(self.val_df, self.tokenizer, 'val', 
                            self.negative_sampler_val, self.task_type,
                             self.max_seq_len, self.sample_data, self.cache_path)
        dataloader = data.DataLoader(dataset,
                                     batch_size=self.val_batch_size,
                                     shuffle=False,
                                     collate_fn=self.data_collator)
        return dataloader

    def _get_test_loader(self):
        dataset = WeaklySupervisedQueryDocumentDataset(self.test_df, self.tokenizer, 'test', 
                             self.negative_sampler_val, self.task_type,
                             self.max_seq_len, self.sample_data, self.cache_path)
        dataloader = data.DataLoader(dataset,
                                     batch_size=self.val_batch_size,
                                     shuffle=False,
                                     collate_fn=self.data_collator)
        return dataloader

class WeaklySupervisedQueryDocumentDataset(data.Dataset):
    """
    Dataset for pointwise learning with <Query,Document> pairs with Weak Supervision.
    The weak supervision is determined by the negative_sampler scores and it is 
    only applied to the negative sampled documents.

    For example, if negative_sampler = bm25, the scores of the negative samples will be
    the bm25 score of that negative sample instead of 0.

    No generative models, e.g. T5, are supported here. 
    
    IMPORTANT:All weak supervision coming from the NS must not include labels 1. 
    Weak supervision has to be (0--1]. This is because in other modules we assume
    everything that is <1 to be label 0, including evaluation.
    """
    def __init__(self, data, tokenizer, data_partition, 
                negative_sampler, task_type, max_seq_len, sample_data,
                cache_path, cache_mode = "memmap"):
        random.seed(42)

        self.data = data
        self.tokenizer = tokenizer
        self.data_partition = data_partition
        self.negative_sampler = negative_sampler
        self.instances = []
        self.task_type = task_type
        self.max_seq_len = max_seq_len
        self.sample_data = sample_data
        self.cache_path = cache_path
        self.cache_mode = cache_mode

        self._group_relevant_documents()
        if self.cache_mode =='memmap':
            self._cache_instances_memmap()
        elif self.cache_mode =='pickle':
            self._cache_instances_pickle()
        del(self.data)

    def _group_relevant_documents(self):
        """
        Since some datasets have multiple relevants per query, we group them to make NS easier.
        """
        query_col = self.data.columns[0]
        num_rel_documents = self.data.shape[0] #datasets have a format of one line per query and doc combination
        self.data = self.data.groupby(query_col).agg(list).reset_index()
        num_queries = self.data.shape[0]
        self.number_instances = (num_queries * self.negative_sampler.num_candidates_samples) + num_rel_documents

    def _cache_instances_pickle(self):
        """
        Loads tensors into memory or creates the dataset when it does not exist already.
        """        
        signature = "weakly_supervised_pointwise_set_{}_n_cand_docs_{}_ns_sampler_{}_seq_max_l_{}_sample_{}_for_{}_using_{}".\
            format(self.data_partition,
                   self.negative_sampler.num_candidates_samples,
                   self.negative_sampler.name,
                   self.max_seq_len,
                   self.sample_data,
                   self.task_type,
                   self.tokenizer.__class__.__name__)
        path = self.cache_path + "/" + signature

        if os.path.exists(path):
            with open(path, 'rb') as f:
                logging.info("Loading instances from {}".format(path))
                self.instances = pickle.load(f)
        else:            
            logging.info("Generating instances with signature {}".format(signature))

            labels = []
            examples = []
            for idx, row in enumerate(tqdm(self.data.itertuples(index=False), total=len(self.data))):
                query = row[0]
                relevant_documents = row[1]
                for relevant_document in relevant_documents:
                    examples.append((query, relevant_document))
                    labels.append(1.0)
                ns_candidates, ns_scores, _, _, _ = self.negative_sampler.sample(query, relevant_documents)                

                for i, ns in enumerate(ns_candidates):
                    examples.append((query, ns))
                    labels.append(ns_scores[i])

            logging.info("Encoding examples using tokenizer.batch_encode_plus().")
            batch_encoding = self.tokenizer(examples, max_length=self.max_seq_len,
                                                      padding="max_length", truncation=True)

            logging.info("Transforming examples to instances format.")
            self.instances = []
            for i in range(len(examples)):
                inputs = {k: batch_encoding[k][i] for k in batch_encoding}                
                feature = InputFeatures(**inputs, label=labels[i])
                self.instances.append(feature)

            for idx in range(3):
                logging.info("Set {} Instance {} query \n\n{}[...]\n".format(self.data_partition, idx, examples[idx][0][0:200]))
                logging.info("Set {} Instance {} document \n\n{}\n".format(self.data_partition, idx, examples[idx][1][0:200]))
                logging.info("Set {} Instance {} features \n\n{}\n".format(self.data_partition, idx, self.instances[idx]))
            with open(path, 'wb') as f:
                pickle.dump(self.instances, f)

        logging.info("Total of {} instances were cached.".format(len(self.instances)))

    def _cache_instances_memmap(self):
        """
        Loads the memset numpy matrices in case they exist otherwise create them.
        """        
        signature = "weakly_supervised_pointwise_set_{}_n_cand_docs_{}_ns_sampler_{}_seq_max_l_{}_sample_{}_for_{}_using_{}.dat".\
            format(self.data_partition,
                   self.negative_sampler.num_candidates_samples,
                   self.negative_sampler.name,
                   self.max_seq_len,
                   self.sample_data,
                   self.task_type,
                   self.tokenizer.__class__.__name__)

        path_input_ids = self.cache_path + "/input_ids_" + signature
        if not os.path.exists(path_input_ids):
            logging.info("Generating instances with signature {}".format(signature))

            #Creating mem_maps
            path_labels = self.cache_path + "/labels_" + signature
            self.mem_maps = {}
            self.mem_maps['labels'] = np.memmap(path_labels, dtype='float', mode='w+', shape=(self.number_instances))
            for k in ['input_ids', 'token_type_ids', 'attention_mask']:
                path_input = self.cache_path + "/{}_".format(k) + signature
                self.mem_maps[k] = np.memmap(path_input, dtype='int', mode='w+', shape=(self.number_instances, self.max_seq_len))

            len_each_batch = min(1000, self.data.shape[0])
            number_batches = self.data.shape[0]//len_each_batch
            logging.info("Batches of size {}.".format(len_each_batch))

            instances_idx = 0
            for idx, chunk in enumerate(tqdm(np.array_split(self.data, number_batches))):
                examples = []
                labels = []
                for row in chunk.itertuples(index=False):
                    query = row[0]
                    relevant_documents = row[1]
                    for relevant_document in relevant_documents:
                        examples.append((query, relevant_document))
                        labels.append(1.0)
                    ns_candidates, ns_scores, _, _, _= self.negative_sampler.sample(query, relevant_documents)                

                    for i, ns in enumerate(ns_candidates):
                        examples.append((query, ns))
                        labels.append(ns_scores[i])

                batch_encoding = self.tokenizer(examples, max_length=self.max_seq_len,
                                                        padding="max_length", truncation=True)
                for k in batch_encoding:
                    np_batch_encoding = np.array(batch_encoding[k])
                    self.mem_maps[k][instances_idx:instances_idx+np_batch_encoding.shape[0]] = np_batch_encoding[:]

                np_labels = np.array(labels)
                self.mem_maps['labels'][instances_idx:instances_idx+np_labels.shape[0]] = np_labels[:]                

                instances_idx+=np_labels.shape[0]

            del(self.mem_maps)

        self.mem_maps = {}
        for k in ['input_ids', 'token_type_ids', 'attention_mask', 'labels']:
            filename = self.cache_path + "/{}_".format(k) + signature
            logging.info("Loading {} from {}".format(k, filename))
            data_shape = (self.number_instances, self.max_seq_len)
            if k == 'labels':
                self.mem_maps[k] = np.memmap(filename, dtype='float', mode='r', shape=(self.number_instances))
            else:
                self.mem_maps[k] = np.memmap(filename, dtype='int', mode='r', shape=data_shape) 

    def __len__(self):
        return self.number_instances

    def __getitem__(self, index):
        if self.cache_mode == 'pickle':
            return self.instances[index]
        elif self.cache_mode == 'memmap':
            instance_data = {}
            for k in self.mem_maps:
                if k != "labels":
                    instance_data[k] = list(self.mem_maps[k][index])
            inputs = {k: instance_data[k] for k in instance_data}
            feature = InputFeatures(**inputs, label=float(self.mem_maps["labels"][index]))
            return feature