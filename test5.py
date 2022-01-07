import torch

a = torch.ones(1,10)

print(a.expand(4,-1))

