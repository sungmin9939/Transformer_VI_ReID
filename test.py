import cv2
import numpy as np
from torch._C import device
from torch.optim import optimizer
from torchvision import transforms
from torchvision.transforms.transforms import Compose
from data_loader import TestData_dum, TestData
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
from utils import AverageMeter, eval_regdb
from tensorboardX import SummaryWriter
import torch.nn as nn



def test(opt):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 128)),
        
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    device = torch.device('cuda:0')
    '''
    gallset = TestData_dum('./datasets/RegDB_01/train_py','T', transform_test)
    queryset = TestData_dum('./datasets/RegDB_01/train_py','V', transform_test)
    '''
    gallset = TestData('./datasets/RegDB_01','gallery', transform_test)
    queryset = TestData('./datasets/RegDB_01','query', transform_test)
    
    
    gall_loader = DataLoader(gallset, batch_size=opt.batch_size, shuffle=False)
    query_loader = DataLoader(queryset, batch_size=opt.batch_size, shuffle=False)

    model = Trans_VIReID(opt).to(device)
    model.eval()

    if os.path.exists(opt.checkpoint):
        model.load_state_dict(torch.load('./checkpoint/exp0_epoch200.pth'))
        
    ptr = 0
    gall_feat = np.zeros((len(gallset), 768))
    gall_feat_att = np.zeros((len(gallset), 768))
    gall_label = np.zeros(len(gallset))
    with torch.no_grad():
        for idx, (img, label) in enumerate(gall_loader):
            batch_num = img.size(0)
            img = Variable(img).to(device)
            feat, feat_att = model(img,img,label,modal=2)
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            gall_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            gall_label[ptr:ptr + batch_num] = label.numpy()
            ptr = ptr + batch_num

    ptr = 0
    query_feat = np.zeros((len(queryset), 768))
    query_feat_att = np.zeros((len(queryset), 768))
    query_label = np.zeros(len(queryset))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input).to(device)
            feat, feat_att = model(input, input, label, modal=1)
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            query_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            query_label[ptr:ptr + batch_num] = label.numpy()
            ptr = ptr + batch_num

    distmat = np.matmul(query_feat, np.transpose(gall_feat))
    distmat_att = np.matmul(query_feat_att, np.transpose(gall_feat_att))

    cmc, mAP, mINP      = eval_regdb(-distmat, query_label, gall_label)
    cmc_att, mAP_att, mINP_att  = eval_regdb(-distmat_att, query_label, gall_label)


    print('rank1: {}'.format(cmc[0]))
    print('mAP: {}'.format(mAP))
    print('mINP: {}'.format(mINP))
    print('rank1_att: {}'.format(cmc_att[0]))
    print('mAP_att: {}'.format(mAP_att))
    print('mINP_att: {}'.format(mINP_att))





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',default='RegDB', type=str)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--decay',default=0.0001, type=float)
    parser.add_argument('--optim',default='Adam', type=str)
    parser.add_argument('--checkpoint',default='./checkpoint/')
    parser.add_argument('--epochs', default=1000)
    parser.add_argument('--log_path', default='./runs/')
    parser.add_argument('--trial',default=0,type=int)

    parser.add_argument('--dim', default=768)
    parser.add_argument('--img_h', default=256, type=int)
    parser.add_argument('--img_w',default=128, type=int)
    parser.add_argument('--patch_size',default=16)
    parser.add_argument('--in_channel',default=3)
    parser.add_argument('--is_train',default=True)
    parser.add_argument('--batch_size',default=16, type=int)
    parser.add_argument('--margin',default=0.5)
    

    opt = parser.parse_args()
    
    test(opt)