import torch
from torch import nn

from models.transformers import PatchEmbedding, TransformerEncoder, Trans_VIReID
from PIL import Image
import numpy as np
import cv2
from torchvision.transforms import Compose, Resize, ToTensor





model = Trans_VIReID(3, 16, 768, 128, 4, 0, 3, 4, 5,4,2,2,2,8)
device = torch.device('cuda:0')
#print(device)
model = model.to(device)

img = Image.open('./cat.jpg')
img2 = Image.open('./dog.jpg')

output = model(img,img2)


