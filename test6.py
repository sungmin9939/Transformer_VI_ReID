from torch.utils.data import DataLoader
from data_loader import SYSUData, RegDBData, TestData
from pprint import pprint


data_path = './datasets/RegDB_01'
dataset = RegDBData(data_path)

pprint(dataset.visible_files[:20])