import torch
from torch import nn

class RegionLoss(nn.Module):
    def __init__(self,):
        super(RegionLoss, self).__init__()
        pass
    def forward(self, pre, gt):
        # input b x 
