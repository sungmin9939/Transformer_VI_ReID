import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn,Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize,ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary
from loss import OriTripletLoss
from transformers import ViTModel, ViTConfig



class MLP(nn.Module):
    def __init__(self, in_feat, hid_feat=None, out_feat=None,
                 dropout=0.):
        super().__init__()
        if not hid_feat:
            hid_feat = in_feat
        if not out_feat:
            out_feat = in_feat
        self.fc1 = nn.Linear(in_feat, hid_feat)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hid_feat, out_feat)
        self.droprateout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.droprateout(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, attention_dropout=0., proj_dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = 1./dim**0.5

        self.qkv = nn.Linear(dim, dim*3, bias=False)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_dropout)
        )

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, c//self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        dot = (q @ k.transpose(-2, -1)) * self.scale
        attn = dot.softmax(dim=-1)
        attn = self.attention_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.out(x)
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768, img_size=128):
        self.patch_size = patch_size
        super().__init__()
        self.transform = Compose([Resize((img_size,img_size)), ToTensor()])
        self.projection = nn.Sequential(
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size)
        )
        self.cls_token = nn.Parameter(torch.randn(1,1,emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size)**2 +1, emb_size))
        self.visible = nn.Parameter(torch.ones((img_size // patch_size)**2 +1, emb_size))
        self.infrared = nn.Parameter(torch.ones((img_size // patch_size)**2 +1, emb_size) + 1)
        

        self.device = torch.device('cuda:0') 
        #print('positions: {}'.format(self.positions.shape))

    def forward(self, x: Tensor, modality) -> Tensor:
        if type(x) != torch.Tensor:
            x = self.transform(x)
            x = x.unsqueeze(0)
            x = x.to(self.device)
        b,_,_,_ = x.shape
        
        x_pt = self.projection(x)
        if torch.count_nonzero(torch.isnan(x_pt)):
            print('projection error')
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x_pt = torch.cat([cls_tokens, x_pt],dim=1)
        x_pt += self.positions
        
        if modality == 'visible':
            x_pt += self.visible
        else:
            x_pt += self.infrared
        return x_pt, x

class Encoder_Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4, drop_rate=0.):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, drop_rate, drop_rate)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim*mlp_ratio, dropout=drop_rate)

    def forward(self, x):
        x1 = self.ln1(x)
        x = x + self.attn(x1)
        x2 = self.ln2(x)
        x = x + self.mlp(x2)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, depth, dim, heads, mlp_ratio=4, drop_rate=0.1):
        super().__init__()
        self.Encoder_Blocks = nn.ModuleList([
            Encoder_Block(dim, heads, mlp_ratio, drop_rate)
            for i in range(depth)])

    def forward(self, x):
        for Encoder_Block in self.Encoder_Blocks:
            x = Encoder_Block(x)
        return x



class Trans_VIReID(nn.Module):
    def __init__(self, opt):
        super().__init__()
        if opt.dataset == "RegDB":
            self.num_classes = 206
        elif opt.dataset == "SYSU":
            self.num_classes = 1
        self.img_h = opt.img_h
        self.img_w = opt.img_w
        self.patch_size = opt.patch_size
        self.patch_overlap = int(self.patch_size/2)
        self.scaled_h = int((self.img_h-self.patch_overlap)/self.patch_overlap)
        self.sclaed_w = int((self.img_w-self.patch_overlap)/self.patch_overlap)
        
        
        self.in_channel = opt.in_channel
        self.dim = opt.dim
        self.is_train = opt.is_train
        
        vit_config = ViTConfig()

        self.disc_encoder = ViTModel(vit_config)
        self.excl_encoder = ViTModel(vit_config)

        self.to_img = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=self.scaled_h, w=self.sclaed_w),
            nn.ConvTranspose2d(self.dim, self.in_channel, kernel_size=(self.patch_size,self.patch_size), stride=(self.patch_overlap,self.patch_overlap))
        )

        self.batchnorm = nn.BatchNorm1d(self.sclaed_w * self.scaled_h+1)
        self.classifier = nn.Linear(self.dim, self.num_classes)

        if self.training:
            self.recon_loss = nn.L1Loss()
            self.id_loss = nn.CrossEntropyLoss()


    def forward(self, x_rgb, x_ir, label=None, modal=0):
        if modal == 0:
            
            disc_rgb = self.disc_encoder(pixel_values=x_rgb, modal=1, interpolate_pos_encoding=True).last_hidden_state
            disc_ir = self.disc_encoder(pixel_values=x_ir, modal=2, interpolate_pos_encoding=True).last_hidden_state

            excl_rgb = self.excl_encoder(x_rgb, modal=1, interpolate_pos_encoding=True).last_hidden_state
            excl_ir = self.excl_encoder(x_ir, modal=2, interpolate_pos_encoding=True).last_hidden_state

            rgb_id = self.classifier(torch.mean(self.batchnorm(disc_rgb), dim=1))
            ir_id = self.classifier(torch.mean(self.batchnorm(disc_ir), dim=1))

            re_rgb = self.to_img(disc_rgb[:,1:] + excl_rgb[:,1:])
            re_ir = self.to_img(disc_ir[:,1:] + excl_ir[:,1:])

            dr_ei = self.to_img(disc_rgb[:,1:] + excl_ir[:,1:])
            di_er = self.to_img(disc_ir[:,1:] + excl_rgb[:,1:])

            recon_loss = self.recon_loss(re_rgb, x_rgb) + self.recon_loss(re_ir, x_ir)
            cross_recon_loss = self.recon_loss(x_rgb, di_er) + self.recon_loss(x_ir, dr_ei)
            id_loss = self.id_loss(rgb_id, label) + self.id_loss(ir_id, label)

            return recon_loss, cross_recon_loss, id_loss, rgb_id, ir_id

        elif modal == 1:
            disc_rgb = self.disc_encoder(pixel_values=x_rgb, modal=1, interpolate_pos_encoding=True).last_hidden_state
            feat = torch.mean(disc_rgb, dim=1)
            feat_att = torch.mean(self.batchnorm(disc_rgb),dim=1)

            return feat, feat_att
        elif modal == 2:
            disc_ir = self.disc_encoder(pixel_values=x_ir, modal=2, interpolate_pos_encoding=True).last_hidden_state
            feat = torch.mean(disc_ir, dim=1)
            feat_att = torch.mean(self.batchnorm(disc_ir),dim=1)

            return feat, feat_att