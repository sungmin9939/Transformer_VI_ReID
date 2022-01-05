import torch
from transformers import ViTFeatureExtractor, ViTModel
from models.transformers_MA import Trans_VIReID
import argparse
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',default='RegDB', type=str)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--decay',default=0.0005, type=float)
parser.add_argument('--optim',default='Adam', type=str)
parser.add_argument('--checkpoint',default='./checkpoint/')
parser.add_argument('--epochs', default=120)
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

model = Trans_VIReID(opt)
model.load_state_dict(torch.load('./checkpoint/start.pth'))

pprint(model.state_dict().keys())





'''
input = torch.randn(1,3,128,256)
input2 = torch.randn(1,3,224,224)
print(model.embeddings.rgb_embeddings.shape)


output = model(input, interpolate_pos_encoding=True, modal=1)
#output2 = model(input2)
print(type(output))
'''
