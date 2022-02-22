from transformers import ViTModel
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torchvision import transforms
import argparse
from models.transformers_MA import Trans_VIReID

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


opt = parser.parse_args()
device = torch.device('cuda:0')


model = Trans_VIReID(opt).to(device)
model.load_state_dict(torch.load('./checkpoint/exp4_epoch60.pth'))


