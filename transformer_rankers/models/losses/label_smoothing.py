import torch.nn.functional as F
import torch.nn as nn
import torch
from IPython import embed

class LabelSmoothingCrossEntropy(nn.Module):
    """ Label Smoothing implementation from https://github.com/huanglianghua
    (https://github.com/pytorch/pytorch/issues/7455).

    Label smoothing is a regularization technique that encourages the model to be
    less confident in its predictions, from "Rethinking the Inception Architecture for Computer Vision"
    (https://arxiv.org/abs/1512.00567).

    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, inputs, target):
        log_prob = F.log_softmax(inputs, dim=-1)
        weight = inputs.new_ones(inputs.size()) * \
            self.smoothing / (inputs.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss.mean()


class WeaklySupervisedLabelSmoothingCrossEntropy(nn.Module):
    """ 
        Weakly Supervised Label Smoothing replaces the uniform distribution from LS
        with the weak supervision signal from the negative sampling procedure.        
    """
    def __init__(self, smoothing=0.1):
        super(WeaklySupervisedLabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, inputs, target):
        """
        target must contain labels 1 when positive and when negative 0 <= t_i <1
        """
        log_prob = F.log_softmax(inputs, dim=-1)

        one_hot_target = target.int().clone().long()

        #there has to be more elegant way of creating the weight mask in pytorch...
        # The normal label smoothing for examples where label = 1
        weight_relevant = inputs.new_ones(inputs.size()) * \
            self.smoothing / (inputs.size(-1) - 1.)
        weight_relevant.scatter_(-1, one_hot_target.unsqueeze(-1), (1. - self.smoothing))
        weight_relevant = weight_relevant * one_hot_target.unsqueeze(-1)

        # Use weak supervision for labels = 0 and logit 0
        weight_weak_supervision = inputs.new_zeros(inputs.size())
        weight_weak_supervision.scatter_(-1, 1-one_hot_target.unsqueeze(-1), 1) # For labels 0 and the negative logit
        weight_weak_supervision = weight_weak_supervision * (target * 1-one_hot_target).unsqueeze(-1)

        # Use (1-weak supervision) for labels = 0 and logit 0
        weight_weak_supervision_pos = inputs.new_zeros(inputs.size())
        weight_weak_supervision_pos.scatter_(-1, one_hot_target.unsqueeze(-1), 1) # For labels 0 and the positive logit
        weight_weak_supervision_pos = weight_weak_supervision_pos * (1-target).unsqueeze(-1)

        weight = weight_relevant + weight_weak_supervision + weight_weak_supervision_pos
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss.mean()