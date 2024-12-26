
# Neighbor Relations Matter in Video Scene Detection
This is an official PyTorch Implementation of **Neighbor Relations Matter in Video Scene Detection**.

## Prepare Dataset
1. Refer to https://github.com/mini-mind/VSMBD
2. The unsupervised learning (self-supeivised learning) settings also refer to https://github.com/mini-mind/VSMBD, 
and the pseudo label generation method is changed to https://github.com/kakaobrain/bassl
3. Generate the dataset by running gen_dataSet in dataloader/supervise_movienet.py; ft_path is the path of ImageNet_shot.pkl extracted by VSMBD. lb_path is the path of the txt file provided by MovieNet. Each txt file name is the IMDB ID of each movie, in which each element marks whether each shot is the end shot (the last shot) of the scene; gph_path is the file path after the sample_20.rar I provided is unzipped. 

## Generate Graph Files
The file in sample_20 corresponds to the sentence $N^{l}_{i}$ signifies the top-k similar shots to the shot-i within a time
window centered on the shot-i with a length of $l$." （In the lower part of Eq.(1)）.

If you are interested in the implementation details of generating this graph, please see issue https://github.com/ExMorgan-Alter/NeighborNet/issues/4.

https://pan.baidu.com/s/1sodKXth7GgHztkHp7tsN8A Code: 2345 

## Supervised Model
- A better learning strategy may have better results. The following checkpoint is what I trained casually.
- ~~Original model using the NeighborAttent and BasicReason modules.~~
  
  ~~mAP:64.1, mIoU:59.2, F1:57.3~~
  
 ~~https://pan.baidu.com/s/1uK0VJ3A2qNte9bOZPcbSWg Code: 2345~~
- An upgraded version of the original model using the MNeighborAttent and MBasicReason modules.

  mAP:72.2, mIoU:57.3, F1:54.0
  
  https://pan.baidu.com/s/1H0fLFzbFuibguq6y587OyQ Code: 1357



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
