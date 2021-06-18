import torch.nn.functional as F
import torch

def nll_loss(output, target):
    # loss for log_softmax
    return F.nll_loss(output, target, weight=torch.tensor([0.32, 0.68]))

def cross_entropy(output, target):
    return F.cross_entropy(output, target)