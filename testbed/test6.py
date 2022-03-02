import os,sys,time
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from einops.layers.torch import Rearrange, Reduce
import torch
from models.resnet import resnet50


a = torch.randn(1,128,768)
re = Rearrange('b (h w) c -> b c h w', h=16, w=8)
print(re(a).shape)