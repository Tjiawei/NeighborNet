# Neighbor Relations Matter in Video Scene Detection
This is an official PyTorch Implementation of **Neighbor Relations Matter in Video Scene Detection**.

## Prepare Dataset
1. Refer to https://github.com/mini-mind/VSMBD
2. The unsupervised learning (self-supeivised learning) settings also refer to https://github.com/mini-mind/VSMBD, 
and the pseudo labels generation method is changed to https://github.com/kakaobrain/bassl
3. Generate the dataset by running gen_dataSet in dataloader/supervise_movienet.py; ft_path is the path of ImageNet_shot.pkl extracted by VSMBD. lb_path is the path of the txt file provided by MovieNet. Each txt file name is the IMDB ID of each movie, which marks whether each shot is the end shot (the last shot) of the scene; gph_path is the file path after the sample_20.rar I provided is unzipped. 

## Generate Graph Files
https://pan.baidu.com/s/1sodKXth7GgHztkHp7tsN8A 提取码: 2345 

## Supervised Model
A better learning strategy may have better results. 

The following checkpoint is what I trained casually.

mAP:64.1, mIoU:59.2, F1:57.3

https://pan.baidu.com/s/1uK0VJ3A2qNte9bOZPcbSWg 提取码: 2345 

## To Do
1. Detailed comments
2. Pretrained (Unsupervised Learning Mode) Weights

## Quote

```
@InProceedings{vsd_neighbor,
    author    = {Tan, Jiawei and Wang, Hongxing and Li, Jiaxin and Ou, Zhilong and Qian, Zhangbin},
    title     = {Neighbor Relations Matter in Video Scene Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {18473-18482}
}
```
