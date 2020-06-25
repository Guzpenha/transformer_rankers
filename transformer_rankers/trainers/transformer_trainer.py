from IPython import embed
from tqdm import tqdm
from transformer_rankers.eval import results_analyses_tools
from transformer_rankers.utils import utils

import logging
import torch
import torch.nn as nn
import torch.optim as optim
import functools
import operator

class TransformerTrainer():
    """ 
    Performs optimization of the neural models

    Logs nDCG at every num_validation_instances epoch. Uses all the visible GPUs.
     
    Args:
        model: the transformer model from transformers library. Both SequenceClassification and ConditionalGeneration are accepted.
        train_loader: pytorch train DataLoader.
        val_loader: pytorch val DataLoader.
        test_loader: pytorch test DataLoader.
        num_ns_eval: number of negative samples for evaluation. Used to accumulate the predictions into lists of the appropiate size.
        task_type: str with either 'classification' or 'generation' for SequenceClassification models and ConditionalGeneration models.
        tokenizer: transformer tokenizer.
        validate_every_epochs: int containing the cycle to calculate validation ndcg.
        num_epochs: int containing the number of epochs to train the model (one epoch = one pass on every instance).
        lr: float containing the learning rate.
        sacred_ex: sacred experiment object to log train metrics. None if not to be used.
        max_grad_norm: float indicating the gradient norm to clip.

    """
    def __init__(self, model, train_loader, val_loader, test_loader,
                 num_ns_eval, task_type, tokenizer, validate_every_epochs,
                 num_validation_instances, num_epochs, lr, sacred_ex,
                 max_grad_norm=0.5):

        self.num_ns_eval = num_ns_eval
        self.task_type = task_type
        self.tokenizer = tokenizer
        self.validate_epochs = validate_every_epochs
        self.num_validation_instances = num_validation_instances
        self.num_epochs = num_epochs
        self.lr = lr
        self.sacred_ex = sacred_ex

        self.num_gpu = torch.cuda.device_count()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info("Device {}".format(self.device))
        logging.info("Num GPU {}".format(self.num_gpu))
        self.model = model.to(self.device)
        if self.num_gpu > 1:
            devices = [v for v in range(self.num_gpu)]
            self.model = nn.DataParallel(self.model, device_ids=devices)

        self.metrics = ['ndcg_cut_10']
        self.best_ndcg=0
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.lr)
        self.max_grad_norm = max_grad_norm

    def fit(self):
        """
        Trains the transformer-based neural ranker.
        """
        logging.info("Total batches per epoch : {}".format(len(self.train_loader)))
        logging.info("Validating every {} epoch.".format(self.validate_epochs))        
        val_ndcg=0
        for epoch in range(self.num_epochs):
            for inputs in tqdm(self.train_loader, desc="Epoch {}".format(epoch), total=len(self.train_loader)):
                self.model.train()

                for k, v in inputs.items():
                    inputs[k] = v.to(self.device)                

                outputs = self.model(**inputs)
                loss = outputs[0] 

                if self.num_gpu > 1:
                    loss = loss.mean() 

                loss.backward()

                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

            if self.validate_epochs > 0 and epoch % self.validate_epochs == 0:
                res, _ = self._validate(loader = self.val_loader)
                val_ndcg = res['ndcg_cut_10']
                if val_ndcg>self.best_ndcg:
                    self.best_ndcg = val_ndcg
                if self.sacred_ex != None:
                    self.sacred_ex.log_scalar('eval_ndcg_10', val_ndcg, epoch+1)                    

            logging.info('Epoch {} val nDCG@10 {:.3f}'.format(epoch + 1, val_ndcg))

    def _validate(self, loader):
        """
        Uses trained model to make predictions on the loader.

        Args:
            loader: the DataLoader containing the set to run the prediction and evaluation.         

        Returns:
            A tuple of (results_dict, logits), containing the evaluation metrics and the
            values of the predictions for every instance. For example:
            ({ 'ndcg_cut_10': 0.5,  'recip_rank': 0.4 },  [[0.01, 0.12], [1.2, 0.9]])
        """

        self.model.eval()
        all_logits = []
        all_labels = []
        for idx, batch in enumerate(loader):
            for k, v in batch.items():
                batch[k] = v.to(self.device)

            with torch.no_grad():
                if self.task_type == "classification":
                    outputs = self.model(**batch)
                    _, logits = outputs[:2]
                    for p in logits[:, 1]:
                        all_logits.append(p.tolist())
                    for l in batch["labels"]:
                        all_labels.append(l.tolist())
                elif self.task_type == "generation":
                    outputs = self.model(**batch)                    
                    _, token_logits = outputs[:2]
                    relevant_token_id = self.tokenizer.encode("relevant")[0]
                    not_relevant_token_id = self.tokenizer.encode("not_relevant")[0]

                    pred_relevant = token_logits[0:, 0 , relevant_token_id]
                    pred_not_relevant = token_logits[0:, 0 , not_relevant_token_id]
                    pred = pred_relevant-pred_not_relevant
                    for p in pred:
                        all_logits.append(p.tolist())                    
                    for l in batch["lm_labels"]:
                        if l[0] == relevant_token_id:
                            label = 1
                        else:
                            label = 0
                        all_labels.append(label)

            if self.num_validation_instances!=-1 and idx > self.num_validation_instances:
                break

        all_labels, all_logits = utils.acumulate_lists(all_labels, all_logits, self.num_ns_eval+1)
        return results_analyses_tools.evaluate_and_aggregate(all_logits, all_labels, self.metrics), all_logits

    def test(self):
        """
        Uses trained model to make predictions on the test loader.

        Returns:
            The logits, i.e. predictions,  for the test instances
        """

        logging.info("Starting evaluation on test.")
        self.num_validation_instances = -1 # no sample on test.
        res, logits = self._validate(self.test_loader)
        for metric, v in res.items():
            logging.info("Test {} : {:4f}".format(metric, v))
        return logits