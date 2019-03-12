# 已有数据集信息

- CNN features: extract *2048-dim* top-layer pooling units of the *101-layered ResNet* from the entire image.
- no image preprocessing such as cropping or augmentation.

## FLO

8189 images from 102 different types of flowers without attribute annotations

- res101.mat ['features', 'image_files', 'labels']
    1. features : (2048, 8189)
    2. images_files : (8189, 1)
    3. labels : (8189, 1)

- att_splits.mat ['att', 'test_seen_loc', 'test_unseen_loc', 'train_loc', 'trainval_loc', 'val_loc']

    1. att : (1024, 102)
    2. test_seen_loc : (1403, 1)
    3. test_unseen_loc : (1155, 1)
    4. train_loc : (5878, 1)
    5. trainval_loc : (5631, 1)
    6. val_loc : (1156, 1)

## CUB

## SUN

## AWA1