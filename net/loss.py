import torch.nn.functional as F
import torch

def isMacOS():
  import platform
  return platform.system() == 'Darwin'

def isCuda():
  import os
  import torch
  device = torch.device("cuda")
  # print(f'typeof(device)={type(device)} device={device}')
  return str(device)=='cuda' and not isMacOS()

def deviceName():
  return 'cuda' if isCuda() else 'cpu'

def nll_loss(output, target):
    # loss for log_softmax
    return F.nll_loss(output, target, weight=torch.tensor([0.32, 0.68]).to(deviceName()))

def cross_entropy(output, target):
    return F.cross_entropy(output, target)