# Cooperative Semantic Segmentation and Image Restoration in Adverse Environmental Conditions

[![Paper](http://img.shields.io/badge/paper-arxiv.1911.00679-B31B1B.svg)](https://arxiv.org/abs/1911.00679)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Authors: **Weihao Xia**, Zhanglin Cheng, Yujiu Yang, Jing-Hao Xue.

Contact: xiawh3@outlook.com


## Preparing Training Data

We conduct experiments on different degradation types, which can be divided into two categories: one is the **Regular Degradation Types**, which contains the six common degradations including watermark, iirgular mask, noise, blur, JPEG compression and chromatic aberrations; the other is the **Adverse Cityscapes** under foggy, rainny and reflection conditions, respectively. With these two categories of degra- dations, we expect to validate the applicability of our method in both general scenes and a specific application–autonomous driving.

For **Regular Degradation Types**,  please refer to [data_generator](https://github.com/xiaweihao/SR-Net/blob/master/data_generator/README.md) for more details.

For **Adverse Cityscapes**, the fog and rain images can be downloaded from the official website of [Cityscapes](https://www.cityscapes-dataset.com/), which are available upon request.

We provide simulation scripts to generate window reflection. Reflection is a frequently-encountered source of image corruption that can arise when shooting through a glass surface. Below is a simulated sample using our provided scripts

<p align="center">
  <img src="/asserts/reflection_simulation.jpg">
</p>

We use pre-trained models to compute the input segmentation or parsing masks, i.e. DeepLabv3 model on Cityscapes dataset, [ResNet50dilated model](https://github.com/CSAILVision/semantic-segmentation-pytorch) for Semantic Segmentation on MIT ADE20K dataset,  [DeepLabV2](https://github.com/kazuto1011/deeplab-pytorch) on COCO-Stuff / PASCAL VOC 2012 dataset, and [BiSeNet](https://github.com/zllrunning/face-parsing.PyTorch) for face parsing on CelebAMask-HQ dataset.

## Training

Coming soon.

## Evaluation

Coming soon.

## Pre-trained Models

Coming soon.

## Results

We conduct experiments on various datasets, which mostly can be found in our [paper](http://arxiv.org/abs/1911.00679). Below are semantic segmentation and image restoration results of watermark and iirgular mask.

<p align="center">
  <img src="/asserts/celeba_wm_results.jpg">
</p>

If you found our paper or code useful, please cite our paper.
```
@article{xia2019adverse,
  author    = {Weihao Xia and Zhanglin Cheng and Yujiu Yang and Jing-Hao Xue},
  title     = {Cooperative Semantic Segmentation and Image Restoration in Adverse Environmental Conditions},
  year      = {2019},
  url       = {http://arxiv.org/abs/1911.00679},
}
```







