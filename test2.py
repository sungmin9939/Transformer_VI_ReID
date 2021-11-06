import torch
from torch import nn

from models.transformers import PatchEmbedding, TransformerEncoder, Trans_VIReID, Generator
from PIL import Image
import numpy as np
import cv2
from torchvision.transforms import Compose, Resize, ToTensor



device = torch.device('cuda:0')

model = Trans_VIReID(3, 16, 768, 128, \
                       mlp_ratio=4,drop_rate=0, num_head=3, depth=4, depth1=5,depth2=4,depth3=2,depth4=2,depth5=2,initial_size=8, num_classes=)
model = model.to(device)
img = Image.open('./cat.jpg')
img2 = Image.open('./dog.jpg')

output = model(img, img2)

'''
generator = Generator(dim=768, heads=3)
generator = generator.to(device)

a = torch.randn((1,768)).to(device)
b = torch.randn((1,768)).to(device)

ab = generator(torch.cat((a,b),dim=1))
'''






