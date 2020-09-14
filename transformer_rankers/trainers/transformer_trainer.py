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
import wandb
import os

class TransformerTrainer():
    """ 
    Performs optimization of the neural models

    Logs nDCG at every num_validation_batches epoch. Uses all the visible GPUs.
     
    Args:
        model: the transformer model from transformers library. Both SequenceClassification and ConditionalGeneration are accepted.
        train_loader: pytorch train DataLoader.
        val_loader: pytorch val DataLoader.
        test_loader: pytorch test DataLoader.
        num_ns_eval: number of negative samples for evaluation. Used to accumulate the predictions into lists of the appropiate size.
        task_type: str with either 'classification' or 'generation' for SequenceClassification models and ConditionalGeneration models.
        tokenizer: transformer tokenizer.
        validate_every_epochs: int containing the number of epochs to calculate validation <validation_metric> when reached. Not used if validate_every_steps is used.
        num_validation_batches: number of validation batches to use for calculating validation <validation_metric> (-1 if all otherwise the number of samples)
        num_epochs: int containing the number of epochs to train the model (one epoch = one pass on every instance).
        lr: float containing the learning rate.
        sacred_ex: sacred experiment object to log train metrics. None if not to be used.
        max_grad_norm: float indicating the gradient norm to clip.
        validate_every_steps: int containing the number of steps (batches) to calculate validation <validation_metric> when reached. (-1 if no logging is required)
        validation_metric: which evaluation metric to use for validation error (e.g. ndcg_cut_10). See transformer_rankers/evaluation for the metrics.

    """
    def __init__(self, model, train_loader, val_loader, test_loader,
                 num_ns_eval, task_type, tokenizer, validate_every_epochs,
                 num_validation_batches, num_epochs, lr, sacred_ex,
                 validate_every_steps=-1, max_grad_norm=0.5, 
                 validation_metric='ndcg_cut_10', num_training_instances=-1):

        self.num_ns_eval = num_ns_eval
        self.task_type = task_type
        self.tokenizer = tokenizer
        self.validation_metric = validation_metric
        self.validate_every_epochs = validate_every_epochs
        self.validate_every_steps = validate_every_steps
        self.num_validation_batches = num_validation_batches
        self.num_epochs = num_epochs
        self.lr = lr
        self.sacred_ex = sacred_ex
        self.num_training_instances=num_training_instances

        self.num_gpu = torch.cuda.device_count()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info("Device {}".format(self.device))
        logging.info("Num GPU {}".format(self.num_gpu))
        self.model = model.to(self.device)
        if self.num_gpu > 1:
            devices = [v for v in range(self.num_gpu)]
            self.model = nn.DataParallel(self.model, device_ids=devices)

        self.best_eval_metric=0
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.lr)
        self.max_grad_norm = max_grad_norm

        #copied from huggingface transformer trainer
        wandb.ensure_configured()
        if wandb.api.api_key is None:
            self._has_wandb = False
        else:
            self._has_wandb = False if os.getenv("WANDB_DISABLED") else True

    def fit(self):
        """
        Trains the transformer-based neural ranker.
        """
        logging.info("Total batches per epoch : {}".format(len(self.train_loader)))
        if self.validate_every_epochs > 0:
            logging.info("Validating every {} epoch.".format(self.validate_every_epochs))
        if self.validate_every_steps > 0:
            logging.info("Validating every {} step.".format(self.validate_every_steps))
        if self._has_wandb:
            wandb.watch(self.model)

        total_steps=0
        total_loss=0
        total_instances=0

        if self.num_training_instances == -1:
            actual_epochs = self.num_epochs
        else:
            instances_in_one_epoch = len(self.train_loader) * self.train_loader.batch_size
            actual_epochs =  -(-self.num_training_instances // instances_in_one_epoch) # rounding up
            logging.info("Actual epochs (rounded up): {}".format(actual_epochs))
        
        for epoch in range(actual_epochs):
            epoch_batches_tqdm = tqdm(self.train_loader, desc="Epoch {}, steps".format(epoch),
                                      total=len(self.train_loader))
            for batch_inputs in epoch_batches_tqdm:
                self.model.train()

                for k, v in batch_inputs.items():
                    batch_inputs[k] = v.to(self.device)

                outputs = self.model(**batch_inputs)
                loss = outputs[0] 

                if self.num_gpu > 1:
                    loss = loss.mean() 

                loss.backward()
                total_loss+=loss.item()

                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
                total_steps+=1
                total_instances+= batch_inputs[k].shape[0]                

                if self.num_training_instances != -1 and total_instances >= self.num_training_instances:
                    logging.info("Reached num_training_instances of {} ({} batches). Early stopping.".format(self.num_training_instances, total_steps))
                    break

                #logging for steps
                is_validation_step = (self.validate_every_steps > 0 and total_steps % self.validate_every_steps == 0)
                if is_validation_step:
                    logits, labels, _ = self.predict(loader = self.val_loader)
                    res = results_analyses_tools.evaluate_and_aggregate(logits, labels, [self.validation_metric])
                    val_metric_res = res[self.validation_metric]
                    if val_metric_res>self.best_eval_metric:
                        self.best_eval_metric = val_metric_res
                    if self.sacred_ex != None:
                        self.sacred_ex.log_scalar(self.validation_metric+"_by_step", val_metric_res, total_steps)
                        self.sacred_ex.log_scalar("avg_loss_by_step", total_loss/total_steps, total_steps)
                    if self._has_wandb:
                        wandb.log({'step': total_steps, self.validation_metric+"_by_step" : val_metric_res})
                        wandb.log({'step': total_steps, "avg_loss_by_step" : total_loss/total_steps})

                    epoch_batches_tqdm.set_description("Epoch {} ({}: {:3f}), steps".format(epoch, self.validation_metric, val_metric_res))

            #logging for epochs
            is_validation_epoch = (self.validate_every_epochs > 0 and epoch % self.validate_every_epochs == 0)
            if is_validation_epoch:
                logits, labels, _ = self.predict(loader = self.val_loader)
                res = results_analyses_tools.evaluate_and_aggregate(logits, labels, [self.validation_metric])
                val_metric_res = res[self.validation_metric]
                if val_metric_res>self.best_eval_metric:
                    self.best_eval_metric = val_metric_res
                if self.sacred_ex != None:
                    self.sacred_ex.log_scalar(self.validation_metric+"_by_epoch", val_metric_res, epoch+1)
                    self.sacred_ex.log_scalar("avg_loss_by_epoch", total_loss/total_steps, epoch+1)
                if self._has_wandb:
                    wandb.log({'epoch': epoch+1, self.validation_metric+"_by_epoch" : val_metric_res})
                    wandb.log({'epoch': epoch+1, "avg_loss_by_epoch" : total_loss/total_steps})
                epoch_batches_tqdm.set_description("Epoch {} ({}: {:3f}), steps".format(epoch, self.validation_metric, val_metric_res))

    def predict(self, loader):
        """
        Uses trained model to make predictions on the loader.

        Args:
            loader: the DataLoader containing the set to run the prediction and evaluation.         

        Returns:
            Matrices (logits, labels, softmax_logits)
        """

        self.model.eval()
        all_logits = []
        all_labels = []
        all_softmax_logits = []
        for idx, batch in tqdm(enumerate(loader), total=len(loader), desc="Predicting"):
            for k, v in batch.items():
                batch[k] = v.to(self.device)

            with torch.no_grad():
                if self.task_type == "classification":
                    outputs = self.model(**batch)
                    _, logits = outputs[:2]
                    all_labels+=batch["labels"].tolist()
                    all_logits+=logits[:, 1].tolist()
                    all_softmax_logits+=torch.softmax(logits, dim=1)[:, 1].tolist()

                elif self.task_type == "generation":
                    outputs = self.model(**batch)                    
                    _, token_logits = outputs[:2]
                    relevant_token_id = self.tokenizer.encode("relevant")[0]
                    not_relevant_token_id = self.tokenizer.encode("not_relevant")[0]

                    pred_relevant = token_logits[0:, 0 , relevant_token_id]
                    pred_not_relevant = token_logits[0:, 0 , not_relevant_token_id]
                    both = torch.stack((pred_relevant, pred_not_relevant))

                    all_logits+=pred_relevant.tolist()
                    all_labels+=[1 if (l[0] == relevant_token_id) else 0 for l in batch["labels"]]
                    all_softmax_logits+=torch.softmax(both, dim=0)[0].tolist()

            if self.num_validation_batches!=-1 and idx > self.num_validation_batches:
                break

        #accumulates per query
        all_labels = utils.acumulate_list_multiple_relevant(all_labels)
        all_logits = utils.acumulate_l1_by_l2(all_logits, all_labels)
        all_softmax_logits = utils.acumulate_l1_by_l2(all_softmax_logits, all_labels)
        return all_logits, all_labels, all_softmax_logits

    def predict_with_uncertainty(self, loader, foward_passes):
        """
        Uses trained model to make predictions on the loader with uncertainty estimations.

        This methods uses MC dropout to get the predicted relevance (mean) and uncertainty (variance)
        by enabling dropout at test time and making K foward passes.

        See "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning"
        https://arxiv.org/abs/1506.02142.

        Args:
            loader: DataLoader containing the set to run the prediction and evaluation.         
            foward_passes: int indicating the number of foward prediction passes for each instance.

        Returns:
            Matrices (logits, labels, softmax_logits, foward_passes_logits, uncertainties):
            The logits (mean) for every instance, labels, softmax_logits (mean) all predictions
            obtained during f_passes (foward_passes_logits) and the uncertainties (variance).
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
        softmax_logits = []
        foward_passes_logits = [[] for i in range(foward_passes)] # foward_passes X queries        
        for idx, batch in tqdm(enumerate(loader), total=len(loader)):
            for k, v in batch.items():
                batch[k] = v.to(self.device)

            with torch.no_grad():
                fwrd_predictions = []
                fwrd_softmax_predictions = []
                if self.task_type == "classification":                    
                    labels+= batch["labels"].tolist()
                    for i, f_pass in enumerate(range(foward_passes)):
                        outputs = self.model(**batch)
                        _, batch_logits = outputs[:2]

                        fwrd_predictions.append(batch_logits[:, 1].tolist())
                        fwrd_softmax_predictions.append(torch.softmax(batch_logits, dim=1)[:, 1].tolist())
                        foward_passes_logits[i]+=batch_logits[:, 1].tolist()
                elif self.task_type == "generation":
                    labels+=[1 if (l[0] == relevant_token_id) else 0 for l in batch["labels"]]
                    for i, f_pass in enumerate(range(foward_passes)):
                        outputs = self.model(**batch)
                        _, token_logits = outputs[:2]
                        pred_relevant = token_logits[0:, 0 , relevant_token_id]
                        pred_not_relevant = token_logits[0:, 0 , not_relevant_token_id]                        
                        both = torch.stack((pred_relevant, pred_not_relevant))

                        fwrd_predictions.append(pred_relevant.tolist())
                        fwrd_softmax_predictions.append(torch.softmax(both, dim=0)[0].tolist())
                        foward_passes_logits[i]+=pred_relevant.tolist()

                logits+= np.array(fwrd_predictions).mean(axis=0).tolist()
                uncertainties += np.array(fwrd_predictions).var(axis=0).tolist()
                softmax_logits += np.array(fwrd_softmax_predictions).mean(axis=0).tolist()
            if self.num_validation_batches!=-1 and idx > self.num_validation_batches:
                break

        #accumulates per query
        labels = utils.acumulate_list_multiple_relevant(labels)
        logits = utils.acumulate_l1_by_l2(logits, labels)
        uncertainties = utils.acumulate_l1_by_l2(uncertainties, labels)
        softmax_logits = utils.acumulate_l1_by_l2(softmax_logits, labels)
        for i, foward_logits in enumerate(foward_passes_logits):
            foward_passes_logits[i] = utils.acumulate_l1_by_l2(foward_logits, labels)

        return logits, labels, softmax_logits, foward_passes_logits, uncertainties

    def test(self):
        """
        Uses trained model to make predictions on the test loader.

        Returns:
            Matrices (logits, labels, softmax_logits)
        """        
        self.num_validation_batches = -1 # no sample on test.
        return self.predict(self.test_loader)
    
    def test_with_dropout(self, foward_passes):
        """
        Uses trained model to make predictions on the test loader using MC dropout as bayesian estimation.

        Returns:
            Matrices (logits, labels, softmax_logits, foward_passes_logits, uncertainties)
        """        
        self.num_validation_batches = -1 # no sample on test.
        return self.predict_with_uncertainty(self.test_loader, foward_passes)