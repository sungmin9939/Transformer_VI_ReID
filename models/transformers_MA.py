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

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        '''
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
        '''



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
        
        #vit_config = ViTConfig()

        self.disc_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.excl_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        #self.disc_encoder.embeddings.rgb_embeddings

        self.to_img = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=self.scaled_h, w=self.sclaed_w),
            nn.ConvTranspose2d(self.dim, self.in_channel, kernel_size=(self.patch_size,self.patch_size), stride=(self.patch_overlap,self.patch_overlap))
        )

        self.batchnorm = nn.BatchNorm1d(self.sclaed_w * self.scaled_h+1)
        self.classifier = nn.Linear(self.dim, self.num_classes)
        self.modality_knowledge = nn.Linear(1, self.dim)
        
        
        self.classifier.apply(weights_init_classifier)
        self.batchnorm.apply(weights_init_kaiming)
        self.to_img.apply(weights_init_kaiming)
        self.modality_knowledge.apply(weights_init_classifier)
        
             
        


    def forward(self, x_rgb, x_ir, modal=0):
        if modal == 0:
            
            disc_rgb = self.disc_encoder(pixel_values=x_rgb, modal=1, interpolate_pos_encoding=True).last_hidden_state
            disc_ir = self.disc_encoder(pixel_values=x_ir, modal=2, interpolate_pos_encoding=True).last_hidden_state

            excl_rgb = self.excl_encoder(x_rgb, modal=1, interpolate_pos_encoding=True).last_hidden_state
            excl_ir = self.excl_encoder(x_ir, modal=2, interpolate_pos_encoding=True).last_hidden_state

            feat_rgb = self.batchnorm(disc_rgb)
            feat_ir = self.batchnorm(disc_ir)

            rgb_id = self.classifier(feat_rgb[:,0])
            ir_id = self.classifier(feat_ir[:,0])

            ##reconstruction part
            re_rgb = self.to_img(disc_rgb[:,1:] + excl_rgb[:,1:])
            re_ir = self.to_img(disc_ir[:,1:] + excl_ir[:,1:])

            dr_ei = self.to_img(disc_rgb[:,1:] + excl_ir[:,1:])
            di_er = self.to_img(disc_ir[:,1:] + excl_rgb[:,1:])

            disc_rgb_hat = self.disc_encoder(pixel_values=dr_ei, modal=2, interpolate_pos_encoding=True).last_hidden_state
            disc_ir_hat = self.disc_encoder(pixel_values=di_er, modal=1, interpolate_pos_encoding=True).last_hidden_state

            excl_rgb_hat = self.excl_encoder(di_er, modal=1, interpolate_pos_encoding=True).last_hidden_state
            excl_ir_hat = self.excl_encoder(dr_ei, modal=2, interpolate_pos_encoding=True).last_hidden_state

            hat_rgb = self.to_img(disc_rgb_hat[:,1:] + excl_rgb_hat[:,1:])
            hat_ir = self.to_img(disc_ir_hat[:,1:] + excl_ir_hat[:,1:])

            ##modality embedding knowledge
            rgb_knowledge = self.modality_knowledge(self.disc_encoder.embeddings.rgb_embeddings)
            ir_knowledge = self.modality_knowledge(self.disc_encoder.embeddings.ir_embeddings)
            
            #make center vector for each identity
            rgb_feat_center = None
            ir_feat_center = None
            
            for chunk in list(torch.split(feat_rgb[:,0], 4)):  #feat_rgb[:,0] (batch_size, 768) chunk (batch_size/4, 768)
                if rgb_feat_center is None:
                    rgb_feat_center = torch.mean(chunk - rgb_knowledge, dim=0)
                else:
                    rgb_feat_center = torch.cat((rgb_feat_center, torch.mean(chunk-rgb_knowledge, dim=0)), dim=0)
                    
            for chunk in list(torch.split(feat_ir[:,0],4)):
                if ir_feat_center is None:
                    ir_feat_center = torch.mean(chunk - ir_knowledge, dim=0)
                else:
                    ir_feat_center = torch.cat((ir_feat_center, torch.mean(chunk-ir_knowledge, dim=0)), dim=0)
            
            #feat_center (batch_size/4, 768)
            
            rgb_knowledged_id = self.classifier(feat_rgb[:,0] - rgb_knowledge)
            ir_knowledged_id = self.classifier(feat_ir[:,0] - ir_knowledge)
            
            
            
            return (disc_rgb, disc_ir),\
                (excl_rgb, disc_ir),\
                (feat_rgb[:,0]-rgb_knowledge, feat_ir[:,0]-ir_knowledge),\
                (rgb_id, ir_id),\
                (re_rgb, re_ir),\
                (di_er, dr_ei),\
                (disc_rgb_hat, disc_ir_hat),\
                (excl_rgb_hat, excl_ir_hat),\
                (hat_rgb, hat_ir),\
                (rgb_feat_center, ir_feat_center),\
                (rgb_knowledged_id, ir_knowledged_id)\
                
            '''
            recon_loss = self.recon_loss(re_rgb, x_rgb) + self.recon_loss(re_ir, x_ir)
            cross_recon_loss = self.recon_loss(x_rgb, di_er) + self.recon_loss(x_ir, dr_ei)
            id_loss = self.id_loss(rgb_id, label) + self.id_loss(ir_id, label)

            return recon_loss, cross_recon_loss, id_loss, rgb_id, ir_id
            '''

        elif modal == 1:
            disc_rgb = self.disc_encoder(pixel_values=x_rgb, modal=1, interpolate_pos_encoding=True).last_hidden_state
            feat = disc_rgb[:,0]
            feat_att =self.classifier(self.batchnorm(disc_rgb)[:,0])

            return feat, feat_att
        elif modal == 2:
            disc_ir = self.disc_encoder(pixel_values=x_ir, modal=2, interpolate_pos_encoding=True).last_hidden_state
            feat = disc_ir[:,0]
            feat_att =self.classifier(self.batchnorm(disc_ir)[:,0])

            return feat, feat_att