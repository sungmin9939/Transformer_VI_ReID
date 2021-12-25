import torch
from models.transformers_MA import TransformerEncoder, PatchEmbedding, Trans_VIReID
from PIL import Image
import argparse


a = torch.isnan(torch.tensor([1, 3, 2]))
if torch.count_nonzero(a):
    print('yes')