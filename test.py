import gc
import numpy as np
from torch._C import device
from torch.optim import optimizer
from torchvision import transforms
from torchvision.transforms.transforms import Compose
from data_loader import TestData_dum, TestData
from models.transformers_MA import Trans_VIReID
from utils import *
import torch.utils.data as data
import torchvision.utils as vutils
import sys
import argparse
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.autograd import Variable
from loss import TripletLoss_WRT
from utils import AverageMeter, eval_regdb
import time
from tensorboardX import SummaryWriter
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image


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
    
    model.load_state_dict(torch.load('./checkpoint/exp6_epoch60.pth'))
        
    ptr = 0
    gall_feat = np.zeros((len(gallset), 768))
    gall_feat_att = np.zeros((len(gallset), 206))
    gall_label = np.zeros(len(gallset))
    gall_names = []
    with torch.no_grad():
        for idx, (img, label, img_name) in enumerate(gall_loader):
            batch_num = img.size(0)
            img = Variable(img).to(device)
            feat, feat_att = model(img,img,modal=2)
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            gall_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            gall_label[ptr:ptr + batch_num] = label.numpy()
            gall_names.extend(list(img_name))
            ptr = ptr + batch_num

    ptr = 0
    query_feat = np.zeros((len(queryset), 768))
    query_feat_att = np.zeros((len(queryset), 206))
    query_label = np.zeros(len(queryset))
    query_names = []
    with torch.no_grad():
        for batch_idx, (input, label, img_name) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input).to(device)
            feat, feat_att = model(input, input, modal=1)
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            query_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            query_label[ptr:ptr + batch_num] = label.numpy()
            query_names.extend(list(img_name))
            ptr = ptr + batch_num

    distmat = np.matmul(query_feat, np.transpose(gall_feat))
    distmat_att = np.matmul(query_feat_att, np.transpose(gall_feat_att))

    cmc, mAP, mINP      = eval_regdb(-distmat, query_label, gall_label, query_names, gall_names)
    #cmc_att, mAP_att, mINP_att  = eval_regdb(-distmat_att, query_label, gall_label, query_names, gall_names)

    

    print('rank1: {}'.format(cmc[0]))
    print('rank5: {}'.format(cmc[4]))
    print('rank10: {}'.format(cmc[9]))
    print('rank20: {}'.format(cmc[19]))
    print('mAP: {}'.format(mAP))
    print('mINP: {}'.format(mINP))
    '''
    print('rank1_att: {}'.format(cmc_att[0]))
    print('mAP_att: {}'.format(mAP_att))
    print('mINP_att: {}'.format(mINP_att))
    '''

def draw_sample_basic_ycbcr():
    gallset = TestData('./datasets/RegDB_01','gallery')
    queryset = TestData('./datasets/RegDB_01','query')
    
    draw_sample_indices = [0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]
    
    images = []
    
    for idx in draw_sample_indices:
        gall = np.array(Image.open(gallset.images[idx])) #gray
        query = np.array(Image.open(queryset.images[idx]).convert('YCbCr')) #YCbCr from RGB
        

        gCbCr = np.zeros(gall.shape, dtype=np.uint8)

        
        gCbCr[:,:,0] = gall[:,:,0]
        gCbCr[:,:,1] = query[:,:,1]
        gCbCr[:,:,2] = query[:,:,2]
        
        img = Image.fromarray(gCbCr)
        img.save('./YCbCr/gcbcr{}.png'.format(idx))
        images.append(torch.Tensor(gCbCr).permute(2,0,1).unsqueeze(0))
     
    image_tensors = torch.cat(images)
    print(image_tensors.shape)
    num_images = image_tensors.size(0)
    
    image_grid = vutils.make_grid(image_tensors, num_images, padding=0, normalize=True, scale_each=True)
    vutils.save_image(image_grid, './YCbCr/all.png',1)
    
       


def draw_sample_basic(opt):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 128)),
        
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    invTrans = transforms.Compose([
        transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
        transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]),
    ])
    

    device = torch.device('cuda:0')
    model = Trans_VIReID(opt).to(device)
    model.eval()
    
    model.load_state_dict(torch.load('./checkpoint/exp6_epoch60.pth'))

    '''
    gallset = TestData_dum('./datasets/RegDB_01/train_py','T', transform_test)
    queryset = TestData_dum('./datasets/RegDB_01/train_py','V', transform_test)
    '''
    
    gallset = TestData('./datasets/RegDB_01','gallery', transform_test)
    queryset = TestData('./datasets/RegDB_01','query', transform_test)
    
    
    gall_loader = DataLoader(gallset, batch_size=opt.batch_size, shuffle=False)
    query_loader = DataLoader(queryset, batch_size=opt.batch_size, shuffle=False)

    draw_sample_indices = [0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]
    
    test_images_display_g = torch.stack(
        [gall_loader.dataset[draw_sample_indices[i]][0] for i in range(len(draw_sample_indices))]
    )
    test_images_display_q = torch.stack(
        [query_loader.dataset[draw_sample_indices[i]][0] for i in range(len(draw_sample_indices))]
    )
    
    num_images = test_images_display_g.size(0)
    
    gall_recon, query_recon, gall_crecon, query_crecon = [],[],[],[]
    
    
    with torch.no_grad():
        for i in range(num_images):
            gall = test_images_display_g[i].unsqueeze(0).to(device)
            query = test_images_display_q[i].unsqueeze(0).to(device)
            
            d_gall = model.disc_encoder(gall, modal=2, interpolate_pos_encoding=True).last_hidden_state
            d_query = model.disc_encoder(query, modal=1, interpolate_pos_encoding=True).last_hidden_state
            
            i_gall = model.excl_encoder(gall, modal=2, interpolate_pos_encoding=True).last_hidden_state
            i_query = model.excl_encoder(query, modal=1, interpolate_pos_encoding=True).last_hidden_state
            
            dg_ig = model.to_img(d_gall[:,1:] + i_gall[:,1:]).cpu()
            dq_iq = model.to_img(d_query[:,1:] + i_query[:,1:]).cpu()
            
            dg_iq = model.to_img(d_gall[:,1:] + i_query[:,1:]).cpu()
            dq_ig = model.to_img(d_query[:,1:] + i_gall[:,1:]).cpu()
            
            del d_gall, d_query, i_gall, i_query
            
            gall_recon.append(dg_ig)
            query_recon.append(dq_iq)
            gall_crecon.append(dq_ig)
            query_crecon.append(dg_iq)
    
    gall_recon, query_recon = torch.cat(gall_recon), torch.cat(query_recon)
    gall_crecon, query_crecon = torch.cat(gall_crecon), torch.cat(query_crecon)
    
    exp = test_images_display_g, gall_recon, gall_crecon, test_images_display_q, query_recon, query_crecon
    
    image_tensor = torch.cat([images for images in exp])
    print(image_tensor.shape)
    image_grid = vutils.make_grid(image_tensor.data, num_images, padding=0, normalize=True, scale_each=True)
    vutils.save_image(image_grid, './test.png',1)
        
        
    
    




def visualize():
    f = open('./test_result/recon/matches.csv','r')
    reader = csv.reader(f)
    lines = list(reader)
    
    
    for i in range(0,len(lines)-1,2):
        fig = plt.figure()
        row, col = 1,6
        
        img_names = lines[i]
        img_matches = list(map(int, lines[i+1]))
        for j in range(len(img_names)):
            ax = fig.add_subplot(row, col, j+1)
            ax.imshow(Image.open(img_names[j]))
            if j == 0:
                continue
            else:
                if img_matches[j-1] == 1:
                    ax.set_title('R@{}'.format(j),color='green')
                else:
                    ax.set_title('R@{}'.format(j),color='red')
            ax.axis('off')
            
        fig.savefig('./test_result/recon/imgs/test{}.png'.format(i))
        plt.close()
            
        
        
        
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',default='RegDB', type=str)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--decay',default=0.0005, type=float)
    parser.add_argument('--optim',default='Adam', type=str)
    parser.add_argument('--checkpoint',default='./checkpoint/')
    parser.add_argument('--epochs', default=70)
    parser.add_argument('--log_path', default='./runs/')
    parser.add_argument('--trial',default=2,type=int)

    parser.add_argument('--dim', default=768)
    parser.add_argument('--img_h', default=256, type=int)
    parser.add_argument('--img_w',default=128, type=int)
    parser.add_argument('--patch_size',default=16)
    parser.add_argument('--in_channel',default=3)
    parser.add_argument('--recon', default=False, type=bool)
    parser.add_argument('--batch_size',default=32, type=int)
    parser.add_argument('--margin',default=0.5)
    

    opt = parser.parse_args()
    
    #draw_sample_basic_ycbcr()
    #draw_sample_basic(opt)
    test(opt)
    #visualize()
    
    
    