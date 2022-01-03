import torch
import torch.nn as nn

device = torch.device('cuda:0')
a = torch.randn(1,768,31,15).to(device)
conv = nn.Conv2d(768,1024,(31,15),1).to(device)
output = conv(a)
print(output.shape)