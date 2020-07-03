from dataclasses import dataclass, field
from typing import Dict, List, Optional

from transformers import DataCollator
from transformers.data.processors.utils import InputFeatures
from transformers.data.data_collator import DefaultDataCollator

from IPython import embed
from tqdm import tqdm
from abc import *

import dataclasses
import functools
import operator
import torch.utils.data as data
import torch
import logging
import random
import os
import pickle

@dataclass
class T2TDataCollator(DataCollator):
    """
    Data collator for generation tasks.
    """
    def collate_batch(self, batch: List) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([torch.tensor(example['input_ids'], dtype=torch.long) for example in batch])
        lm_labels = torch.stack([torch.tensor(example['target_ids'], dtype=torch.long) for example in batch])
        lm_labels[lm_labels[:, :] == 0] = -100
        attention_mask = torch.stack([torch.tensor(example['attention_mask'], dtype=torch.long) for example in batch])
        decoder_attention_mask = torch.stack([torch.tensor(example['target_attention_mask'], dtype=torch.long) for example in batch])

        return {
            'input_ids': input_ids, 
            'attention_mask': attention_mask,
            'lm_labels': lm_labels, 
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
            self.data_collator = DefaultDataCollator()
        elif self.task_type == "generation":
            self.data_collator = T2TDataCollator()
            special_tokens_dict = {
                'additional_special_tokens': ['not_relevant']
            }
            self.tokenizer.add_special_tokens(special_tokens_dict)

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
                                     collate_fn=self.data_collator.collate_batch)
        return dataloader

    def _get_val_loader(self):
        dataset = QueryDocumentDataset(self.val_df, self.tokenizer, 'val', 
                            self.negative_sampler_val, self.task_type,
                             self.max_seq_len, self.sample_data, self.cache_path)
        dataloader = data.DataLoader(dataset,
                                     batch_size=self.val_batch_size,
                                     shuffle=False,
                                     collate_fn=self.data_collator.collate_batch)
        return dataloader

    def _get_test_loader(self):
        dataset = QueryDocumentDataset(self.test_df, self.tokenizer, 'test', 
                             self.negative_sampler_val, self.task_type,
                             self.max_seq_len, self.sample_data, self.cache_path)
        dataloader = data.DataLoader(dataset,
                                     batch_size=self.val_batch_size,
                                     shuffle=False,
                                     collate_fn=self.data_collator.collate_batch)
        return dataloader

class QueryDocumentDataset(data.Dataset):
    def __init__(self, data, tokenizer, data_partition, 
                negative_sampler, task_type, max_seq_len, sample_data,
                cache_path):
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
        signature = "set_{}_n_cand_docs_{}_ns_sampler_{}_seq_max_l_{}_sample_{}_for_{}".\
            format(self.data_partition,
                   self.negative_sampler.num_candidates_samples,
                   self.negative_sampler.name,
                   self.max_seq_len,
                   self.sample_data,
                   self.task_type)
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
                ns_candidates, _, _ = self.negative_sampler.sample(query, relevant_documents)                
                for ns in ns_candidates:
                    examples.append((query, ns))

            logging.info("Encoding examples using tokenizer.batch_encode_plus().")
            batch_encoding = self.tokenizer.batch_encode_plus(examples, 
                max_length=self.max_seq_len, pad_to_max_length=True)
            
            if self.task_type == "generation": 
                target_encodings = self.tokenizer.batch_encode_plus(labels, 
                    pad_to_max_length=True, max_length=10)
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
                    inputs = {**inputs, **targets}
                if self.task_type == "classification":
                    feature = InputFeatures(**inputs, label=labels[i])
                else:
                    feature = inputs
                self.instances.append(feature)            

            for idx in range(3):
                logging.info("Set {} Instance {} query \n\n{}[...]\n".format(self.data_partition, idx, examples[idx][0][0:200]))
                logging.info("Set {} Instance {} document \n\n{}\n".format(self.data_partition, idx, examples[idx][1][0:200]))
                logging.info("Set {} Instance {} features \n\n{}\n".format(self.data_partition, idx, self.instances[idx]))
            with open(path, 'wb') as f:
                pickle.dump(self.instances, f)

        logging.info("Total of {} instances were cached.".format(len(self.instances)))

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        return self.instances[index]