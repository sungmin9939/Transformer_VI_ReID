import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn,Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize,ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary
from loss import OriTripletLoss


class MLP(nn.Module):
    def __init__(self, in_feat, hid_feat=None, out_feat=None,
                 dropout=0.):
        super().__init__()
        if not hid_feat:
            hid_feat = in_feat
        if not out_feat:
            out_feat = in_feat
        self.fc1 = nn.Linear(in_feat, hid_feat)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hid_feat, out_feat)
        self.droprateout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.droprateout(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, attention_dropout=0., proj_dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = 1./dim**0.5

        self.qkv = nn.Linear(dim, dim*3, bias=False)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_dropout)
        )

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, c//self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        dot = (q @ k.transpose(-2, -1)) * self.scale
        attn = dot.softmax(dim=-1)
        attn = self.attention_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.out(x)
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768, img_size=128):
        self.patch_size = patch_size
        super().__init__()
        self.transform = Compose([Resize((img_size,img_size)), ToTensor()])
        self.projection = nn.Sequential(
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size)
        )
        self.cls_token = nn.Parameter(torch.randn(1,1,emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size)**2 +1, emb_size))
        self.visible = nn.Parameter(torch.ones((img_size // patch_size)**2 +1, emb_size))
        self.infrared = nn.Parameter(torch.ones((img_size // patch_size)**2 +1, emb_size)) + 1)
        

        self.device = torch.device('cuda:0') 
        #print('positions: {}'.format(self.positions.shape))

    def forward(self, x: Tensor, modality) -> Tensor:
        if type(x) != torch.Tensor:
            x = self.transform(x)
            x = x.unsqueeze(0)
            x = x.to(self.device)
        b,_,_,_ = x.shape
        
        x_pt = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x_pt = torch.cat([cls_tokens, x_pt],dim=1)
        x_pt += self.positions
        if modality == 'visible':
            x_pt += self.visible
        else:
            x_pt += self.infrared
        return x_pt, x
