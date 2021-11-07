from data_loader import RegDBData
from torchvision import transforms
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from models.transformers import PatchEmbedding
transform_train_list = [
        transforms.Resize((128,128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
transform = transforms.Compose(transform_train_list)
dataset = RegDBData('./datasets/RegDB_01', transform)

dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
v, t, l = next(iter(dataloader))
print(v.shape)
print(l)

patchemb = PatchEmbedding()
patch = patchemb(v)
print(patch[0].shape)


def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = tensor.permute(1,2,0)
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    
    return Image.fromarray(tensor)
