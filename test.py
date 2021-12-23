from os import rename
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch import nn,Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize,ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary


x = torch.randn(3,4)
x = x.unflatten(2,6)