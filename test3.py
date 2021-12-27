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
from utils import AverageMeter, eval_regdb
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

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_test = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        normalize,
    ])


    device = torch.device('cuda:0')

    #Initiate dataset
    if opt.dataset == 'RegDB':
        data_path = './datasets/RegDB_01'
        dataset = RegDBData(data_path, transform)
    elif opt.dataset == 'sysu':
        data_path = './datasets/SYSU-MM01'
        dataset = SYSUData(data_path, transform)
    gallset = TestData('./datasets/RegDB_01','gallery', transform_test)
    queryset = TestData('./datasets/RegDB_01','query', transform_test)
    
    
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
    gall_loader = DataLoader(gallset, batch_size=opt.batch_size, shuffle=False)
    query_loader = DataLoader(queryset, batch_size=opt.batch_size, shuffle=False)

    
    criterion_tri = TripletLoss_WRT()

    if opt.optim == 'sgd':
        optimizer = optim.SGD(model.parameters())
    elif opt.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.decay)
    
    for i in range(opt.epochs):
        trainloader = tqdm(dataloader)
        model.train()

        for idx, (rgb, ir, label) in enumerate(trainloader):
            rgb = Variable(rgb).to(device)
            ir = Variable(ir).to(device)
            label = Variable(label).to(device)
            


            
            recon_loss, cross_recon_loss, id_loss, rgb_id, ir_id = model(rgb, ir, label, modal=0)
            
            

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
        writer.add_scalar('id_loss', loss_id.avg, i)
        print(
            'epoch: {}\ntrain_loss: {}\nrecon_loss: {}\ncross_recon_loss: {}\ntri_loss: {}\nid_loss: {}'.format(i, loss_train.avg, loss_recon.avg, loss_crecon.avg, loss_tri.avg, loss_id.avg)
        )
        if i % 10 == 0:
            torch.save(model.state_dict(), './checkpoint/exp{}_epoch{}.pth'.format(opt.trial, i))

        #test
        print("Testing model Accuracy...")
        model.eval()
        ptr = 0
        gall_feat = np.zeros((len(gallset),65*768))
        gall_feat_att = np.zeros((len(gallset),65*768))
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
        query_feat = np.zeros((len(queryset), 65*768))
        query_feat_att = np.zeros((len(queryset), 65*768))
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
    parser.add_argument('--decay',default=0.0001, type=float)
    parser.add_argument('--optim',default='Adam', type=str)
    parser.add_argument('--checkpoint',default='./checkpoint/')
    parser.add_argument('--epochs', default=1000)
    parser.add_argument('--log_path', default='./runs/')
    parser.add_argument('--trial',default=2)

    parser.add_argument('--depth', default=5)
    parser.add_argument('--dim', default=768)
    parser.add_argument('--heads', default=4)
    parser.add_argument('--mlp_ratio', default=4)
    parser.add_argument('--drop_rate', default=0.1, type=float)
    parser.add_argument('--img_size',default=128)
    parser.add_argument('--patch_size',default=16)
    parser.add_argument('--in_channel',default=3)
    parser.add_argument('--is_train',default=True)
    parser.add_argument('--batch_size',default=64, type=int)
    parser.add_argument('--margin',default=0.5)
    

    opt = parser.parse_args()



    main(opt)