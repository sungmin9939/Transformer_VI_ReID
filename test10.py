from importlib.metadata import requires
import torch
import numpy as np

device = torch.device('cuda:0')

a = torch.FloatTensor(1).to(device)
b = torch.FloatTensor(1).to(device)
c = a+b
print(a)