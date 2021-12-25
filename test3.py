import cv2
import numpy as np
from torch._C import device
from torch.optim import optimizer
from torchvision import transforms
from torchvision.transforms.transforms import Compose
from data_loader import SYSUData, RegDBData, TestData
from models.transformers_MA import Trans_VIReID
from utils import *
import torch.utils.data as data
import sys
import argparse
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.autograd import Variable
from loss import TripletLoss_WRT





def main(opt):


    transform_train_list = [
        transforms.Resize((128,128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    device = torch.device('cuda:0')
    transform = Compose(transform_train_list)
    if opt.dataset == 'RegDB':
        data_path = './datasets/RegDB_01'
        dataset = RegDBData(data_path, transform)
    elif opt.dataset == 'sysu':
        data_path = './datasets/SYSU-MM01'
        dataset = SYSUData(data_path, transform)
    
    model = Trans_VIReID(opt).to(device)

    if os.path.exists(opt.checkpoint):
        pass



    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
    
    criterion_tri = TripletLoss_WRT()

    if opt.optim == 'sgd':
        optimizer = optim.SGD(model.parameters())
    elif opt.optim == 'Adam':
        optimizer = optim.Adam(model.parameters())
    
    for i in range(opt.epochs):
        trainloader = tqdm(dataloader)
        model.train()

        for idx, (rgb, ir, label) in enumerate(trainloader):
            rgb = Variable(rgb).to(device)
            ir = Variable(ir).to(device)
            label = Variable(label).to(device)

            
            loss, rgb_id, ir_id = model(rgb, ir, label)

            tri_loss = criterion_tri(torch.cat((rgb_id, ir_id), dim=0), torch.cat((label, label)))

            total_loss = loss + tri_loss[0]

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()






    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',default='RegDB', type=str)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--optim',default='Adam', type=str)
    parser.add_argument('--checkpoint',default='./checkpoint/')
    parser.add_argument('--epochs', default=1000)

    parser.add_argument('--depth', default=5)
    parser.add_argument('--dim', default=768)
    parser.add_argument('--heads', default=4)
    parser.add_argument('--mlp_ratio', default=4)
    parser.add_argument('--drop_rate', default=0.1, type=float)
    parser.add_argument('--img_size',default=128)
    parser.add_argument('--patch_size',default=16)
    parser.add_argument('--in_channel',default=3)
    parser.add_argument('--is_train',default=True)
    parser.add_argument('--batch_size',default=8)
    parser.add_argument('--margin',default=0.5)
    

    opt = parser.parse_args()



    main(opt)