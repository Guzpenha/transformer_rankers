from IPython import embed
from tqdm import tqdm
from transformer_rankers.eval import results_analyses_tools
from transformer_rankers.utils import utils

import logging
import torch
import torch.nn as nn
import numpy as np
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
        num_validation_instances: number of validation instances to use (-1 if all)
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
                logits, labels = self.predict(loader = self.val_loader)
                res = results_analyses_tools.evaluate_and_aggregate(logits, labels, ['ndcg_cut_10'])
                val_ndcg = res['ndcg_cut_10']
                if val_ndcg>self.best_ndcg:
                    self.best_ndcg = val_ndcg
                if self.sacred_ex != None:
                    self.sacred_ex.log_scalar('eval_ndcg_10', val_ndcg, epoch+1)                    

            logging.info('Epoch {} val nDCG@10 {:.3f}'.format(epoch + 1, val_ndcg))

    def predict(self, loader):
        """
        Uses trained model to make predictions on the loader.

        Args:
            loader: the DataLoader containing the set to run the prediction and evaluation.         

        Returns:
            A tuple of (logits, labels). For example:
            ([[0.01, 0.12], [1.2, 0.9]])
        """

        self.model.eval()
        all_logits = []
        all_labels = []
        for idx, batch in tqdm(enumerate(loader), total=len(loader)):
            for k, v in batch.items():
                batch[k] = v.to(self.device)

            with torch.no_grad():
                if self.task_type == "classification":
                    outputs = self.model(**batch)
                    _, logits = outputs[:2]
                    all_labels+=batch["labels"].tolist()
                    all_logits+=logits[:, 1].tolist()

                elif self.task_type == "generation":
                    outputs = self.model(**batch)                    
                    _, token_logits = outputs[:2]
                    relevant_token_id = self.tokenizer.encode("relevant")[0]
                    not_relevant_token_id = self.tokenizer.encode("not_relevant")[0]

                    pred_relevant = token_logits[0:, 0 , relevant_token_id]
                    pred_not_relevant = token_logits[0:, 0 , not_relevant_token_id]
                    pred = pred_relevant-pred_not_relevant                    

                    all_logits+=pred.tolist()
                    all_labels+=[1 if (l[0] == relevant_token_id) else 0 for l in batch["lm_labels"]]

            if self.num_validation_instances!=-1 and idx > self.num_validation_instances:
                break

        #accumulates per query
        all_labels = utils.acumulate_list_multiple_relevant(all_labels)
        all_logits = utils.acumulate_l1_by_l2(all_logits, all_labels)
        return all_logits, all_labels

    def predict_with_uncertainty(self, loader, foward_passes):
        """
        Uses trained model to make predictions on the loader with uncertainty estimations.

        This methods uses MC dropout to get the predicted relevance (mean) and uncertainty (variance)
        by enabling dropout at test time and making K foward passes. Use foward_passes=1 for deterministic
        point-estimate relevance predictions.

        See "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning"
        https://arxiv.org/abs/1506.02142.

        Args:
            loader: DataLoader containing the set to run the prediction and evaluation.         
            foward_passes: int indicating the number of foward prediction passes for each instance.

        Returns:
            A triplet of (logits, uncertainties, labels), the values of the predictions for every instance
            the uncertainty (variance) and the labels. For example:
            ([[0.01, 0.12], [1.2, 0.9], [0.05, 0.5]], [[1,0], [1,0]])
        """
        def enable_dropout(model):
            for module in model.modules():
                if module.__class__.__name__.startswith('Dropout'):
                    module.train()

        self.model.eval()
        enable_dropout(self.model)

        if self.task_type == "generation":
            relevant_token_id = self.tokenizer.encode("relevant")[0]
            not_relevant_token_id = self.tokenizer.encode("not_relevant")[0]

        logits = []
        labels = []
        uncertainties = []
        for idx, batch in tqdm(enumerate(loader), total=len(loader)):
            for k, v in batch.items():
                batch[k] = v.to(self.device)

            with torch.no_grad():
                if self.task_type == "classification":                    
                    labels+= batch["labels"].tolist()
                    fwrd_predictions = []
                    for f_pass in range(foward_passes):
                        outputs = self.model(**batch)
                        _, batch_logits = outputs[:2]
                        fwrd_predictions.append(batch_logits[:, 1].tolist())
                elif self.task_type == "generation":
                    labels+=[1 if (l[0] == relevant_token_id) else 0 for l in batch["lm_labels"]]
                    fwrd_predictions = []
                    for f_pass in range(foward_passes):
                        outputs = self.model(**batch)
                        _, token_logits = outputs[:2]
                        pred_relevant = token_logits[0:, 0 , relevant_token_id]
                        pred_not_relevant = token_logits[0:, 0 , not_relevant_token_id]
                        pred = pred_relevant-pred_not_relevant
                        fwrd_predictions.append(pred.tolist())

                logits+= np.array(fwrd_predictions).mean(axis=0).tolist()
                uncertainties += np.array(fwrd_predictions).var(axis=0).tolist()
            if self.num_validation_instances!=-1 and idx > self.num_validation_instances:
                break

        #accumulates per query
        labels = utils.acumulate_list_multiple_relevant(labels)
        logits = utils.acumulate_l1_by_l2(logits, labels)
        uncertainties = utils.acumulate_l1_by_l2(uncertainties, labels)
        return logits, uncertainties, labels

    def test(self):
        """
        Uses trained model to make predictions on the test loader.

        Returns:
            A tuple of (logits, labels)
        """        
        self.num_validation_instances = -1 # no sample on test.
        return self.predict(self.test_loader)
    
    def test_with_dropout(self, foward_passes):
        """
        Uses trained model to make predictions on the test loader using MC dropout as bayesian estimation.

        Returns:
            A triplet of (logits, uncertainties, labels)
        """        
        self.num_validation_instances = -1 # no sample on test.
        return self.predict_with_uncertainty(self.test_loader, foward_passes)