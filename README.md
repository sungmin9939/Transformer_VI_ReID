# Transformer_VI_ReID


1. patch embeddings => modality embedding 추가 완료
2. patch embeddings => patch overlapping 추가 완료


forward process
1. input image
2. transform the input image(resize(256,128), normalize, random horizontal filp & erasing...)
3. patchembedding the input image
3.1 add modality embedding of size (1, num_patches +1, emb_size) according to modality of input

