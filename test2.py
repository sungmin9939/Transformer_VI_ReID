import torch

a = torch.tensor([[[1,2,3],[4,5,6]],[[1,2,3],[4,5,6]]])
b = torch.tensor([[1,1,1],[1,1,1]])
a += b
print(a)