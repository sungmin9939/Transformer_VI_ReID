from os import rename
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn,Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize,ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary


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
        self.device = torch.device('cuda:0') 
        #print('positions: {}'.format(self.positions.shape))

    def forward(self, x: Tensor) -> Tensor:
        if type(x) != torch.Tensor:
            x = self.transform(x)
            x = x.unsqueeze(0)
            x = x.to(self.device)
        b,_,_,_ = x.shape
        
        x_pt = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x_pt = torch.cat([cls_tokens, x_pt],dim=1)
        x_pt += self.positions
        return x_pt, x


class ImgPatches(nn.Module):
    def __init__(self, input_channel=3, dim=768, patch_size=4):
        super().__init__()
        self.patch_embed = nn.Conv2d(input_channel, dim,
                                     kernel_size=patch_size, stride=patch_size)

    def forward(self, img):
        patches = self.patch_embed(img).flatten(2).transpose(1, 2)
        return patches

def UpSampling(x, H, W):
        B, N, C = x.size()
        assert N == H*W
        x = x.permute(0, 2, 1)
        x = x.view(-1, C, H, W)
        x = nn.PixelShuffle(2)(x)
        B, C, H, W = x.size()
        x = x.view(-1, C, H*W)
        x = x.permute(0,2,1)
        return x, H, W

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
    def __init__(self, depth, dim, heads, mlp_ratio=4, drop_rate=0.):
        super().__init__()
        self.Encoder_Blocks = nn.ModuleList([
            Encoder_Block(dim, heads, mlp_ratio, drop_rate)
            for i in range(depth)])

    def forward(self, x):
        for Encoder_Block in self.Encoder_Blocks:
            x = Encoder_Block(x)
        return x

class Generator(nn.Module):
    """docstring for Generator"""
    def __init__(self, depth1=5, depth2=4, depth3=2, depth4=2, depth5=1, initial_size=8, dim=384, heads=4, mlp_ratio=4, drop_rate=0.):#,device=device):
        super(Generator, self).__init__()

        #self.device = device
        self.initial_size = initial_size
        self.dim = dim
        self.depth1 = depth1
        self.depth2 = depth2
        self.depth3 = depth3
        self.depth4 = depth4
        self.depth5 = depth5
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.droprate_rate =drop_rate

        self.mlp = nn.Linear(self.dim*2, (self.initial_size ** 2) * self.dim)

        self.positional_embedding_1 = nn.Parameter(torch.zeros(1, (self.initial_size**2), self.dim))
        self.positional_embedding_2 = nn.Parameter(torch.zeros(1, (self.initial_size*2)**2, self.dim//4))
        self.positional_embedding_3 = nn.Parameter(torch.zeros(1, (self.initial_size*4)**2, self.dim//16))
        self.positional_embedding_4 = nn.Parameter(torch.zeros(1, (self.initial_size*8)**2, self.dim//64))
        self.positional_embedding_5 = nn.Parameter(torch.zeros(1, (self.initial_size*16)**2, self.dim//256))

        self.TransformerEncoder_encoder1 = TransformerEncoder(depth=self.depth1, dim=self.dim,heads=self.heads, mlp_ratio=self.mlp_ratio, drop_rate=self.droprate_rate)
        self.TransformerEncoder_encoder2 = TransformerEncoder(depth=self.depth2, dim=self.dim//4, heads=self.heads, mlp_ratio=self.mlp_ratio, drop_rate=self.droprate_rate)
        self.TransformerEncoder_encoder3 = TransformerEncoder(depth=self.depth3, dim=self.dim//16, heads=self.heads, mlp_ratio=self.mlp_ratio, drop_rate=self.droprate_rate)
        self.TransformerEncoder_encoder4 = TransformerEncoder(depth=self.depth4, dim=self.dim//64, heads=self.heads, mlp_ratio=self.mlp_ratio, drop_rate=self.droprate_rate)
        self.TransformerEncoder_encoder5 = TransformerEncoder(depth=self.depth5, dim=self.dim//256, heads=self.heads, mlp_ratio=self.mlp_ratio, drop_rate=self.droprate_rate)


        #self.linear = nn.Sequential(nn.Conv2d(self.dim//16, 3, 1, 1, 0))

    def forward(self, code):

        x = self.mlp(code).view(-1, self.initial_size ** 2, self.dim)

        x = x + self.positional_embedding_1
        H, W = self.initial_size, self.initial_size
        x = self.TransformerEncoder_encoder1(x)

        x,H,W = UpSampling(x,H,W) 
        x = x + self.positional_embedding_2
        x = self.TransformerEncoder_encoder2(x)

        x,H,W = UpSampling(x,H,W)
        x = x + self.positional_embedding_3
        x = self.TransformerEncoder_encoder3(x)

        x,H,W = UpSampling(x,H,W)
        x = x + self.positional_embedding_4
        x = self.TransformerEncoder_encoder4(x)

        x,H,W = UpSampling(x,H,W)
        x = x + self.positional_embedding_5
        x = self.TransformerEncoder_encoder5(x)

        x = x.permute(0,2,1).view(-1, self.dim//256, H,W)
        #x = self.linear(x.permute(0, 2, 1).view(-1, self.dim//16, H, W))

        return x

class Discriminator(nn.Module):
    def __init__(self, diff_aug, image_size=32, patch_size=4, input_channel=3, num_classes=1,
                 dim=384, depth=7, heads=4, mlp_ratio=4,
                 drop_rate=0.):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError('Image size must be divisible by patch size.')
        num_patches = (image_size//patch_size) ** 2
        self.diff_aug = diff_aug
        self.patch_size = patch_size
        self.depth = depth
        # Image patches and embedding layer
        self.patches = ImgPatches(input_channel, dim, self.patch_size)

        # Embedding for patch position and class
        self.positional_embedding = nn.Parameter(torch.zeros(1, num_patches+1, dim))
        self.class_embedding = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.positional_embedding, std=0.2)
        nn.init.trunc_normal_(self.class_embedding, std=0.2)

        self.droprate = nn.Dropout(p=drop_rate)
        self.TransfomerEncoder = TransformerEncoder(depth, dim, heads,
                                      mlp_ratio, drop_rate)
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        b = x.shape[0]
        cls_token = self.class_embedding.expand(b, -1, -1)

        x = self.patches(x)
        x = torch.cat((cls_token, x), dim=1)
        x += self.positional_embedding
        x = self.droprate(x)
        x = self.TransfomerEncoder(x)
        x = self.norm(x)
        x = self.out(x[:, 0])
        return x

class Trans_VIReID(nn.Module):
    def __init__(self, in_channel, patch_size, emb_size, img_size, mlp_ratio, drop_rate, num_head, depth, depth1, depth2, depth3, depth4, depth5, initial_size, num_classes):
        super().__init__()
        
        self.in_channel = in_channel
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.img_size = img_size
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.num_head = num_head
        self.depth = depth
        self.depth1 = depth1
        self.depth2 = depth2
        self.depth3 = depth3
        self.depth4 = depth4
        self.depth5 = depth5
        self.initial_size = initial_size
        self.num_classes = num_classes

        #loss functions
        self.l1 = nn.L1Loss()
        

        self.patch_emb = PatchEmbedding(self.in_channel, self.patch_size, self.emb_size, self.img_size)

        self.disc_encoder_rgb = TransformerEncoder(self.depth, self.emb_size, self.num_head, self.mlp_ratio, self.drop_rate)
        self.excl_encoder_rgb = TransformerEncoder(self.depth, self.emb_size, self.num_head, self.mlp_ratio, self.drop_rate)

        self.disc_encoder_ir = TransformerEncoder(self.depth, self.emb_size, self.num_head, self.mlp_ratio, self.drop_rate)
        self.excl_encoder_ir = TransformerEncoder(self.depth, self.emb_size, self.num_head, self.mlp_ratio, self.drop_rate)

        self.generator = Generator(self.depth1,self.depth2,self.depth3,self.depth4, self.depth5, self.initial_size, self.emb_size, self.num_head, self.mlp_ratio, self.drop_rate)
        self.classifier = nn.Linear(self.emb_size, self.num_classes)
        '''
        self.generator_rgb = Generator()
        self.generator_ir = Generator()
        '''

    

    def forward(self, x_rgb, x_ir, train=False):

        #embbeding images to patches    
        patch_x_rgb,x_rgb = self.patch_emb(x_rgb)
        patch_x_ir, x_ir = self.patch_emb(x_ir)

        #extract discriminative features from transformers
        disc_rgb = self.disc_encoder_rgb(patch_x_rgb)
        disc_ir = self.disc_encoder_ir(patch_x_ir)
        
        emb_r = self.classifier(disc_rgb.mean(dim=1))
        emb_i = self.classifier(disc_ir.mean(dim=1))

        #(FOR TRAINING)
        if train:
            #extract excluded features from transformers
            excl_rgb = self.excl_encoder_rgb(patch_x_rgb)
            excl_ir = self.excl_encoder_ir(patch_x_ir)
            #generating counterpart images while preserving identity infromation
            x_r2i = self.generator(torch.cat((disc_rgb.mean(dim=1), excl_ir.mean(dim=1)), dim=1)) ## disc_rgb + excl_ir
            x_i2r = self.generator(torch.cat((disc_ir.mean(dim=1), excl_rgb.mean(dim=1)), dim=1)) ## disc_ir + excl_rgb

            x_r2r = self.generator(torch.cat((disc_rgb.mean(dim=1), excl_rgb.mean(dim=1)), dim=1)) ## disc_rgb + excl_rgb
            x_i2i = self.generator(torch.cat((disc_ir.mean(dim=1), excl_ir.mean(dim=1)), dim=1)) ## disc_ir + excl_ir

            x_r2i = self.patch_emb(x_r2i)
            x_i2r = self.patch_emb(x_i2r)

            disc_i2r = self.disc_encoder_rgb(x_i2r)
            excl_i2r = self.excl_encoder_rgb(x_i2r)

            disc_r2i = self.disc_encoder_ir(x_r2i)
            excl_r2i = self.excl_encoder_ir(x_r2i)


            x_r_cycle = self.generator(torch.cat((disc_r2i.mean(dim=1), excl_i2r.mean(dim=1)), dim=1)) ## disc_r2i + excl_i2r
            x_i_cycle = self.generator(torch.cat((disc_i2r.mean(dim=1), excl_r2i.mean(dim=1)), dim=1)) ## disc_i2r + excl_r2i

            #calculating losses...

            #image reconstruction loss(same)
            recon_r = self.recon_criterion(x_rgb, x_r2r)
            recon_i = self.recon_criterion(x_ir, x_i2i)
            same_recon = recon_i + recon_r
            #image reconstruction loss(cross)
            recon_r = self.recon_criterion(x_rgb, x_i2r)
            recon_i = self.recon_criterion(x_ir, x_r2i)
            cross_recon = recon_r + recon_i
            #image reconstruction loss(cycle)
            recon_r = self.recon_criterion(x_rgb, x_r_cycle)
            recon_i = self.recon_criterion(x_ir, x_i_cycle)
            cycle_recon = recon_i + recon_r
            #code reconstruction loss
            recon_code_r = self.recon_criterion(disc_i2r, disc_rgb)
            recon_code_i = self.recon_criterion(disc_r2i, disc_ir)
            code_recon = recon_code_r + recon_code_i

            #triplet loss


        return 0


    def recon_criterion(input, target):
        return torch.mean(torch.abs(input - target))

    