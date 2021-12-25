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
from utils import AverageMeter
from tensorboardX import SummaryWriter



def main(opt):

    


    #Loss recoder
    loss_recon = AverageMeter()
    loss_crecon = AverageMeter()
    loss_tri = AverageMeter()
    loss_id = AverageMeter()
    loss_train = AverageMeter()

    #Image Transformation
    transform_train_list = [
        transforms.Resize((128,128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    transform = Compose(transform_train_list)


    device = torch.device('cuda:0')

    #Initiate dataset
    if opt.dataset == 'RegDB':
        data_path = './datasets/RegDB_01'
        dataset = RegDBData(data_path, transform)
    elif opt.dataset == 'sysu':
        data_path = './datasets/SYSU-MM01'
        dataset = SYSUData(data_path, transform)
    
    #Initiate loss logging directory
    suffix = opt.dataset
    suffix = suffix + '_{}_trial{}_batch{}'.format(opt.optim, opt.trial, opt.batch_size)

    log_dir = opt.log_path + '/' + suffix + '/'

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    
    writer = SummaryWriter(log_dir)

    #make model
    model = Trans_VIReID(opt).to(device)

    if os.path.exists(opt.checkpoint):
        pass



    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
    
    criterion_tri = TripletLoss_WRT()

    if opt.optim == 'sgd':
        optimizer = optim.SGD(model.parameters())
    elif opt.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    
    for i in range(opt.epochs):
        trainloader = tqdm(dataloader)
        model.train()

        for idx, (rgb, ir, label) in enumerate(trainloader):
            rgb = Variable(rgb).to(device)
            ir = Variable(ir).to(device)
            label = Variable(label).to(device)
            if torch.count_nonzero(torch.isnan(rgb)):
                print('input nan!')


            
            recon_loss, cross_recon_loss, id_loss, rgb_id, ir_id = model(rgb, ir, label)

            tri_loss = criterion_tri(torch.cat((rgb_id, ir_id), dim=0), torch.cat((label, label)))

            total_loss = recon_loss + cross_recon_loss + id_loss + tri_loss[0]

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loss_recon.update(recon_loss.item(), rgb.size(0)*2)
            loss_crecon.update(cross_recon_loss.item(), rgb.size(0)*2)
            loss_id.update(id_loss, rgb.size(0)*2)
            loss_tri.update(tri_loss[0], rgb.size(0)*2)
            loss_train.update(total_loss, rgb.size(0)*2)

        writer.add_scalar('train_loss', loss_train.avg, i)
        writer.add_scalar('recon_loss', loss_recon.avg, i)
        writer.add_scalar('cross_recon_loss', loss_crecon.avg, i)
        writer.add_scalar('tri_loss', loss_tri.avg, i)
        writer.add_scalar('loss_id', loss_id.avg, i)









    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',default='RegDB', type=str)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--optim',default='Adam', type=str)
    parser.add_argument('--checkpoint',default='./checkpoint/')
    parser.add_argument('--epochs', default=1000)
    parser.add_argument('--log_path', default='./runs/')
    parser.add_argument('--trial',default=0)

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