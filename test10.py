import torch
from models.transformers_MA import TransformerEncoder, PatchEmbedding
from PIL import Image

device = torch.device('cuda:0')
model = TransformerEncoder(5, 768, 4).to(device)

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