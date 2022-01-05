import torch
import torch.nn as nn

input = torch.randn(8,466,768)

conv1 = nn.Conv1d(466, 233, 1, 1)
conv2 = nn.Conv1d(233,1,1,1)

output1 = conv1(input)
output2 = conv2(output1)

print(output1.shape)
print(output2.shape)
output2 = output2.squeeze(1)
print(output2.shape)