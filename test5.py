import torch
import torch.nn as nn

a = torch.randn(9,100)
b = torch.split(a, 4)

print(b[2].shape)
