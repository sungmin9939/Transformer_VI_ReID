import torch
from models.transformers_MA import TransformerEncoder, PatchEmbedding, Trans_VIReID
from PIL import Image
import argparse
import torch.nn as nn

a = nn.Parameter(torch.ones(5,5))
b = torch.randn(5,5)
print(a+b)