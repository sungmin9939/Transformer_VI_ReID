import torch
from transformers import ViTFeatureExtractor, ViTModel

model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

input = torch.randn(1,3,128,256)
input2 = torch.randn(1,3,224,224)
print(model.embeddings.rgb_embeddings.shape)


output = model(input, interpolate_pos_encoding=True, modal=1)
#output2 = model(input2)
print(output.last_hidden_state.shape)
