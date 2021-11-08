from models.transformers import Generator
import torch


g = Generator()

device = torch.device('cuda:0')

g = g.to(device)