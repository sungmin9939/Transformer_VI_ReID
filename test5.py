from torch._C import dtype
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from loss import OriTripletLoss

input = torch.randn(3, 5, requires_grad=True)
target = torch.tensor([0,2,3])
ce_loss = nn.CrossEntropyLoss()
loss = ce_loss(input, target)

