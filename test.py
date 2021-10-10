import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn,Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize,ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary

img = Image.open('./cat.jpg')

transform = Compose([Resize((224,224)), ToTensor()])
x = transform(img)
x = x.unsqueeze(0)
print(x.shape)

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768, img_size=224):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size)
        )
        self.cls_token = nn.Parameter(torch.randn(1,1,emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size)**2 +1, emb_size)) 
        print('positions: {}'.format(self.positions.shape))

    def forward(self, x: Tensor) -> Tensor:
        b,_,_,_ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_tokens, x],dim=1)
        x += self.positions
        return x

print(PatchEmbedding()(x).shape)