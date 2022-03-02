import os,sys,time
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
from models.transformers_MA import CMTStem
from models.resnet import visible_module, thermal_module, ResidualDecoder
from einops.layers.torch import Rearrange, Reduce

device = torch.device("cuda:0")


x = torch.randn(1,3,256,128).to(device)

stem = CMTStem(3, 3, 32,2).to(device)
vis = visible_module(share_net=4).to(device)
thm = thermal_module(share_net=4)

'''
decoder1 = ResidualDecoder(1, 4, 768, 32,3,0.1,res_norm='bn').to(device)
decoder2 = ResidualDecoder(2, 4, 768, 256, 3, 0.1, 7,3,res_norm='bn').to(device)

projection1 = nn.Conv2d(32,768,(16,16), (8,8)).to(device)
projection2 = nn.Conv2d(256,768,(16,16), (8,8)).to(device)
'''

output2 = vis(x)

print(output2.shape)

