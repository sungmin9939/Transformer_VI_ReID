import cv2
import numpy as np
from torchvision import transforms
from torchvision.transforms.transforms import Compose
from data_loader import SYSUData, RegDBData, TestData
from models.transformers import Trans_VIReID
from utils import *
import torch.utils.data as data
import sys
import argparse
import torch.optim as optim
import torch
from torch.utils.data import DataLoader





def main(opt):


    transform_train_list = [
        transforms.Resize((128,128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    
    transform = Compose(transform_train_list)
    if opt.dataset == 'RegDB':
        data_path = './datasets/RegDB_01'
        num_classes = 206
        dataset = RegDBData(data_path, transform)
    elif opt.dataset == 'sysu':
        data_path = './datasets/SYSU-MM01'
    
    model = Trans_VIReID(3, 16, 768, 128, \
                       mlp_ratio=4,drop_rate=0, num_head=3, depth=4, depth1=5,depth2=4,depth3=2,depth4=2,depth5=2,initial_size=8, num_classes=206)
    
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
    if opt.optim == 'sgd':
        optmizer = optim.SGD(
            model.parameters()
        )
    


    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',default='RegDB')
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--optim',default='sgd', type=str)
    parser.add_argument('--batch_size',default=8)
    parser.add_argument('--img_size',default=128)
    

    opt = parser.parse_args()



    main(opt)