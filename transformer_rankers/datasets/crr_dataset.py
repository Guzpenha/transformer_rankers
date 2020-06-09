from transformers import BertTokenizer, PreTrainedTokenizer 
from transformers.data.processors.utils import InputFeatures
from transformers.data.data_collator import DefaultDataCollator

from IPython import embed
from tqdm import tqdm
from abc import *

import functools
import operator
import torch.utils.data as data
import torch
import logging
import random
import os
import pickle

class AbstractDataloader(metaclass=ABCMeta):
    def __init__(self, args, train_df, val_df, test_df, tokenizer, negative_sampler_train, negative_sampler_val):
        self.args = args
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.negative_sampler_train = negative_sampler_train
        self.negative_sampler_val = negative_sampler_val

        self.num_gpu = torch.cuda.device_count()

        if args.max_gpu != -1:
            self.num_gpu = args.max_gpu
        self.actual_train_batch_size = self.args.train_batch_size \
                                       * max(1, self.num_gpu)
        logging.info("Train instances per batch {}".
                     format(self.actual_train_batch_size))

    @abstractmethod
    def get_pytorch_dataloaders(self):
        pass

class CRRDataLoader(AbstractDataloader):
    def __init__(self, args, train_df, val_df, test_df, tokenizer, negative_sampler_train, negative_sampler_val):
        super().__init__(args, train_df, val_df, test_df, tokenizer, negative_sampler_train, negative_sampler_val)
        special_tokens_dict = {
            'additional_special_tokens': ['[UTTERANCE_SEP]', '[TURN_SEP]']
        }
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.data_collator = DefaultDataCollator()

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader

    def _get_train_loader(self):
        dataset = CRRDataset(self.args, self.train_df,
                             self.tokenizer,'train', self.negative_sampler_train)
        dataloader = data.DataLoader(dataset,
                                     batch_size=self.actual_train_batch_size,
                                     shuffle=True,
                                     collate_fn=self.data_collator.collate_batch)
        return dataloader

    def _get_val_loader(self):
        dataset = CRRDataset(self.args, self.val_df,
                             self.tokenizer, 'val', self.negative_sampler_val)
        dataloader = data.DataLoader(dataset,
                                     batch_size=self.args.val_batch_size,
                                     shuffle=False,
                                     collate_fn=self.data_collator.collate_batch)
        return dataloader

    def _get_test_loader(self):
        dataset = CRRDataset(self.args, self.test_df,
                             self.tokenizer, 'test', self.negative_sampler_val)
        dataloader = data.DataLoader(dataset,
                                     batch_size=self.args.val_batch_size,
                                     shuffle=False,
                                     collate_fn=self.data_collator.collate_batch)
        return dataloader

class CRRDataset(data.Dataset):
    def __init__(self, args, data, tokenizer, data_partition, negative_sampler):
        random.seed(42)

        self.args = args
        self.data = data
        self.tokenizer = tokenizer
        self.data_partition = data_partition
        self.negative_sampler = negative_sampler
        self.instances = []

        self._cache_instances()

    def _cache_instances(self):        
        signature = "set_{}_n_cand_docs_{}_ns_sampler_{}_seq_max_l_{}_sample_{}".\
            format(self.data_partition,
                   self.negative_sampler.num_candidates_samples,
                   self.negative_sampler.name,
                   self.args.max_seq_len,
                   self.args.sample_data)
        path = self.args.data_folder + self.args.task + "/" + signature

        if os.path.exists(path):
            with open(path, 'rb') as f:
                logging.info("Loading instances from {}".format(path))
                self.instances = pickle.load(f)
        else:            
            logging.info("Generating instances with signature {}".format(signature))
            labels = [[1] + ([0] * (self.negative_sampler.num_candidates_samples))] * self.data.shape[0]
            labels = functools.reduce(operator.iconcat, labels, []) #flattening

            examples = []
            for idx, row in enumerate(tqdm(self.data.itertuples(index=False))):
                context = row[0]
                relevant_response = row[1]
                examples.append((context, relevant_response))
                ns_candidates = self.negative_sampler.sample(context, relevant_response)
                for ns in ns_candidates:
                    examples.append((context, ns))

            batch_encoding = self.tokenizer.batch_encode_plus(examples, 
                max_length=self.args.max_seq_len, pad_to_max_length=True)

            self.instances = []
            for i in range(len(examples)):
                inputs = {k: batch_encoding[k][i] for k in batch_encoding}
                feature = InputFeatures(**inputs, label=labels[i])
                self.instances.append(feature)

            for idx in range(3):
                logging.info("Set {} Instance {} context \n\n{}[...]\n".format(self.data_partition, idx, examples[i][0][0:50]))
                logging.info("Set {} Instance {} response \n\n{}\n".format(self.data_partition, idx, examples[i][1][0:50]))                             
                logging.info("Set {} Instance {} features \n\n{}\n".format(self.data_partition, idx, self.instances[i]))
                logging.info("Set {} Instance {} reconstructed input \n\n{}\n".format(self.data_partition, idx,
                    self.tokenizer.convert_ids_to_tokens(self.instances[i].input_ids)))
            with open(path, 'wb') as f:
                pickle.dump(self.instances, f)

        logging.info("Total of {} instances were cached.".format(len(self.instances)))

    def __len__(self):        
        return len(self.instances)

    def __getitem__(self, index):
        return self.instances[index]