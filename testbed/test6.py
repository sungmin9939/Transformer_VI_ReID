import os,sys,time
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from einops.layers.torch import Rearrange, Reduce
import torch
from models.resnet import resnet50
from models.transformers_MA import ClassBlock

classifier = ClassBlock(768,206)

a = torch.randn(4,768)
output = classifier(a)

print(output.shape)