from torch.utils.data import DataLoader
from data_loader import SYSUData, RegDBData, TestData, IdentitySampler
from pprint import pprint
import numpy as np
import random
from torchvision import transforms

transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256,128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomErasing(),
        
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 128)),
        
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])




data_path = './datasets/RegDB_01'
dataset = RegDBData(data_path, transform_train)
sampler = IdentitySampler(dataset, batch_size=16)

gallset = TestData('./datasets/RegDB_01','gallery', transform_test)
queryset = TestData('./datasets/RegDB_01','query', transform_test)


'''
step = 0
for i in sampler:
    if step > 36:
        break
    print(i)
    step += 1
'''    
img, l = queryset.__getitem__(1)


print(l)