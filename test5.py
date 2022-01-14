import torch
import torch.nn as nn

a = torch.randn(8,199,768)
b = torch.randn(8,199,768)

cri = nn.L1Loss()

c = cri(a,b)

print(c)