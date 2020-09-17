import torch.nn.functional as F
import torch.nn as nn
from IPython import embed

class LabelSmoothingCrossEntropy(nn.Module):
    """ Label Smoothing implementation from https://github.com/seominseok0429
    (https://github.com/seominseok0429/label-smoothing-visualization-pytorch/blob/master/utils.py).

    Label smoothing is a regularization technique that encourages the model to be
    less confident in its predictions, from "Rethinking the Inception Architecture for Computer Vision"
    (https://arxiv.org/abs/1512.00567).

    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, inputs, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(inputs, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()