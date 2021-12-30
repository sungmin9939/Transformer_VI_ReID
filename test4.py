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
parser.add_argument('--decay',default=0.0001, type=float)
parser.add_argument('--optim',default='Adam', type=str)
parser.add_argument('--checkpoint',default='./checkpoint/')
parser.add_argument('--epochs', default=1000)
parser.add_argument('--log_path', default='./runs/')
parser.add_argument('--trial',default=3,type=int)

parser.add_argument('--dim', default=768)
parser.add_argument('--heads', default=4)
parser.add_argument('--mlp_ratio', default=4)
parser.add_argument('--drop_rate', default=0.1, type=float)
parser.add_argument('--img_h', default=256, type=int)
parser.add_argument('--img_w',default=128, type=int)
parser.add_argument('--patch_size',default=16)
parser.add_argument('--in_channel',default=3)
parser.add_argument('--is_train',default=True)
parser.add_argument('--batch_size',default=128, type=int)
parser.add_argument('--margin',default=0.5)


opt = parser.parse_args()
device = torch.device('cuda:0')


model = Trans_VIReID(opt).to(device)
model.load_state_dict(torch.load('./checkpoint/test.pth'))
#backbone = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

x = torch.randn(2,3,256,128).to(device)
y = torch.randn(1,3,256,128).to(device)

feat, feat_att = model(x,y,modal=1)
print(feat.shape)