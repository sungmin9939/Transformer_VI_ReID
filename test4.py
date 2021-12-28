import torch
import torch.nn as nn

conv = nn.Conv2d(3, 768, kernel_size=(16,16), stride=(8,8))
x = torch.randn(1,3,256,128)
output = conv(x).flatten(2).transpose(1,2)


print(output.shape)

conv2 = nn.ConvTranspose2d(768,3,kernel_size=(16,16),stride=(8,8))
to_img = nn.Sequential(
    
)
input = conv2(output)

print(input.shape)