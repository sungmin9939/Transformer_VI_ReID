import torch
from models.transformers_MA import TransformerEncoder, PatchEmbedding, Trans_VIReID
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',default='RegDB')
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--optim',default='sgd', type=str)



parser.add_argument('--depth', default=5)
parser.add_argument('--dim', default=768)
parser.add_argument('--heads', default=4)
parser.add_argument('--mlp_ratio', default=4)
parser.add_argument('--drop_rate', default=0)
parser.add_argument('--num_classes', default=412)
parser.add_argument('--img_size',default=128)
parser.add_argument('--patch_size',default=16)
parser.add_argument('--in_channel',default=3)
parser.add_argument('--is_train',default=True)
parser.add_argument('--batch_size',default=8)
parser.add_argument('--margin',default=0.5)


opt = parser.parse_args()

device = torch.device('cuda:0')
model = Trans_VIReID(opt)


img = Image.open('./cat.jpg')
img2 = Image.open('./dog.jpg')

embedder = PatchEmbedding().to(device)
patches = embedder(img, 'visible')

output = model(patches[0])
print(output.shape)

output = output[:,1:,:].transpose(1,2)
print(output.shape)
output = output.unflatten(2, (8,8))
print(output.shape)