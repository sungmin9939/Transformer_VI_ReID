from torch.utils.data import DataLoader
from data_loader import SYSUData, RegDBData, TestData, IdentitySampler
from pprint import pprint
import numpy as np
import random

data_path = './datasets/RegDB_01'
dataset = RegDBData(data_path)
sampler = IdentitySampler(dataset, batch_size=16)


loader = DataLoader(dataset, batch_size=16, sampler=sampler)
step = 0
for i in sampler:
    if step > 36:
        break
    print(i)
    step += 1