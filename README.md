# Transformer_VI_ReID

* patch embeddings => modality embedding 추가 완료

* patch embeddings => patch overlapping 추가 완료

* dataloader => each identity contain 4 rgb, 4 ir (one call of __getitem__)

* modality aware loss 



* ## forward process

  * input image
  * transform the input image(resize(256,128), normalize, random horizontal filp & erasing...)
  * patchembedding the input image
    3.1 add modality embedding of size (1, num_patches +1, emb_size) according to modality of input

