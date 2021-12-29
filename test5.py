import torch
import torch.nn as nn


bn = nn.BatchNorm1d(466)
a = torch.randn(8,466,768)

b = bn(a)
print(b.shape)