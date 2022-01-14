import cv2
import numpy as np
from torch._C import device
from torch.optim import optimizer
from torchvision import transforms
from torchvision.transforms.transforms import Compose
from data_loader import SYSUData, RegDBData, TestData, IdentitySampler
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



def main(opt):

    


    #Loss recoder
    loss_recon = AverageMeter()
    loss_crecon = AverageMeter()
    loss_tri = AverageMeter()
    loss_id = AverageMeter()
    loss_train = AverageMeter()

    #Image Transformation
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


    device = torch.device('cuda:0')

    #Initiate dataset
    if opt.dataset == 'RegDB':
        data_path = './datasets/RegDB_01'
        dataset = RegDBData(data_path, transform_train)
        sampler = IdentitySampler(dataset, opt.batch_size)
    elif opt.dataset == 'sysu':
        data_path = './datasets/SYSU-MM01'
        dataset = SYSUData(data_path, transform_train)
    gallset = TestData('./datasets/RegDB_01','gallery', transform_test)
    queryset = TestData('./datasets/RegDB_01','query', transform_test)
    
    
    #Initiate loss logging directory
    suffix = opt.dataset
    suffix = suffix + '_{}_trial{}_batch{}'.format(opt.optim, opt.trial, opt.batch_size)

    log_dir = opt.log_path + '/' + suffix + '/'

    if not os.path.isdir(log_dir):
        print(log_dir)
        os.makedirs(log_dir)
    
    writer = SummaryWriter(log_dir)

    #make model
    np_model = Trans_VIReID(opt).to(device)

    '''
    if os.path.exists(opt.checkpoint):
        np_model.load_state_dict(torch.load('./checkpoint/start.pth'))
    '''
    model = nn.DataParallel(np_model, device_ids=[0, 1, 2, 3])

    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, sampler=sampler)
    gall_loader = DataLoader(gallset, batch_size=opt.batch_size, shuffle=False)
    query_loader = DataLoader(queryset, batch_size=opt.batch_size, shuffle=False)

    
    criterion_tri = TripletLoss_WRT()
    criterion_id = nn.CrossEntropyLoss()
    criterion_recon = nn.L1Loss()
    pdist = nn.PairwiseDistance(2)

    if opt.optim == 'sgd':
        optimizer = optim.SGD(model.parameters())
    elif opt.optim == 'Adam':
        optimizer = optim.AdamW([{'params': np_model.disc_encoder.parameters(), 'lr':0.0001},
                                 {'params': np_model.excl_encoder.parameters(), 'lr':0.0001},
                                 {'params': np_model.to_img.parameters()},
                                 {'params': np_model.batchnorm.parameters()},
                                 {'params': np_model.classifier.parameters()}], lr=opt.lr, weight_decay=opt.decay)
        #optimizer = optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.decay)
        
    
    for i in range(opt.epochs):
        trainloader = tqdm(dataloader)
        np_model.train()
        model.train()
        
        if i == 15 or i == 30:
            for g in optimizer.param_groups:
                g['lr'] *= 0.1
        
        

        for idx, (rgb, ir, label) in enumerate(trainloader):
            rgb = Variable(rgb).to(device)
            ir = Variable(ir).to(device)
            label = Variable(label).to(device)
            

            out, out_feat, out_id, out_re, out_cross, out_center, out_aware_id = model(rgb, ir)
            
            
            tri_loss = criterion_tri(torch.cat((out[0][:,0], out[1][:,0]),dim=0), torch.cat((label, label)))
            
            '''
            ##MAE loss for out_feat##
            mae_loss = 0
            rgb_feat_split = list(torch.split(out_feat[0], 4))
            ir_feat_split = list(torch.split(out_feat[1], 4))
            for j in range(out_center[0].size(0)):
                rgb_distance = pdist(rgb_feat_split[j], out_center[0][j].expand(4,-1))
                ir_distance = pdist(ir_feat_split[j], out_center[1][j].expand(4,-1))
                total_dist = torch.cat((rgb_distance, ir_distance),dim=0)
                
                mae_loss += torch.sum(torch.log(1 + torch.exp(total_dist)))
                
            maid_loss = criterion_id(out_aware_id[0], label) + criterion_id(out_aware_id[1], label)
            '''
                 
                    
                
                
            
            
            id_loss = criterion_id(out_id[0], label) + criterion_id(out_id[1], label)
            
            recon_loss = criterion_recon(rgb, out_re[0]) + criterion_recon(ir, out_re[1])
            cross_recon_loss = criterion_recon(rgb, out_cross[0]) + criterion_recon(ir, out_cross[1])
            
            total_loss = tri_loss[0] + id_loss + recon_loss + cross_recon_loss #+ mae_loss + maid_loss

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
        writer.add_scalar('id_loss', loss_id.avg, i)
        print(
            'epoch: {}\ntrain_loss: {}\nrecon_loss: {}\ncross_recon_loss: {}\ntri_loss: {}\nid_loss: {}'.format(i, loss_train.avg, loss_recon.avg, loss_crecon.avg, loss_tri.avg, loss_id.avg)
        )
        if i % 20 == 0 and i != 0:
            torch.save(model.state_dict(), './checkpoint/exp{}_epoch{}.pth'.format(opt.trial, i))

        #test
        print("Testing model Accuracy...")
        np_model.eval()
        model.eval()
        ptr = 0
        gall_feat = np.zeros((len(gallset),768))
        gall_feat_att = np.zeros((len(gallset),206))
        gall_label = np.zeros(len(gallset))
        with torch.no_grad():
            for idx, (img, label) in enumerate(gall_loader):
                batch_num = img.size(0)
                img = Variable(img).to(device)
                feat, feat_att = np_model(img,img,modal=2)
                gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
                gall_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
                gall_label[ptr:ptr + batch_num] = label.numpy()
                ptr = ptr + batch_num
    
        ptr = 0
        query_feat = np.zeros((len(queryset), 768))
        query_feat_att = np.zeros((len(queryset), 206))
        query_label = np.zeros(len(queryset))
        with torch.no_grad():
            for batch_idx, (input, label) in enumerate(query_loader):
                batch_num = input.size(0)
                input = Variable(input).to(device)
                feat, feat_att = np_model(input, input, modal=1)
                query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
                query_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
                query_label[ptr:ptr + batch_num] = label.numpy()
                ptr = ptr + batch_num

        distmat = np.matmul(query_feat, np.transpose(gall_feat))
        distmat_att = np.matmul(query_feat_att, np.transpose(gall_feat_att))

        cmc, mAP, mINP      = eval_regdb(-distmat, query_label, gall_label)
        cmc_att, mAP_att, mINP_att  = eval_regdb(-distmat_att, query_label, gall_label)

        writer.add_scalar('rank1', cmc[0], i)
        writer.add_scalar('mAP', mAP, i)
        writer.add_scalar('mINP', mINP, i)
        writer.add_scalar('rank1_att', cmc_att[0], i)
        writer.add_scalar('mAP_att', mAP_att, i)
        writer.add_scalar('mINP_att', mINP_att, i)

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
    parser.add_argument('--decay',default=0.0005, type=float)
    parser.add_argument('--optim',default='Adam', type=str)
    parser.add_argument('--checkpoint',default='./checkpoint/')
    parser.add_argument('--epochs', default=70)
    parser.add_argument('--log_path', default='./runs/')
    parser.add_argument('--trial',default=0,type=int)

    parser.add_argument('--dim', default=768)
    parser.add_argument('--img_h', default=256, type=int)
    parser.add_argument('--img_w',default=128, type=int)
    parser.add_argument('--patch_size',default=16)
    parser.add_argument('--in_channel',default=3)
    parser.add_argument('--is_train',default=True)
    parser.add_argument('--batch_size',default=32, type=int)
    parser.add_argument('--margin',default=0.5)
    

    opt = parser.parse_args()



    main(opt)