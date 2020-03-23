# Cooperative Semantic Segmentation and Image Restoration in Adverse Environmental Conditions

[![Paper](http://img.shields.io/badge/paper-arxiv.1911.00679-B31B1B.svg)](https://arxiv.org/abs/1911.00679)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Authors: **Weihao Xia**, Zhanglin Cheng, Yujiu Yang, Jing-Hao Xue.

Contact: xiawh3@outlook.com


## Datasets

Reflection is a frequently-encountered source of image corruption that can arise when shooting through a glass surface. Below is a simulated sample using our provided scripts

<p align="center">
  <img src="/data_generator/reflection/reflection_simulation.jpg">
</p>

You can download the fog and rain images from the official website of [Cityscapes](https://www.cityscapes-dataset.com/), which are available upon request.

The [Matlab scripts](https://github.com/xialeiliu/RankIQA/tree/master/data/data_generator) of RankIQA may be useful to generate degraded images. You can generate most types of the degradation mentioned in [TID2013](https://www.sciencedirect.com/science/article/pii/S0923596514001490) (17/24, apart from 3, 4, 12, 13, 20, 21, 24). 

No. | Type of distortion | Supported | No. | Type of distortion | Supported
------ | ------ | ------ | ------ | ------ | ------ 
1 | Additive Gaussian noise | Y | 13 | JPEG2000 transmission errors | N 
2 | Additive noise in color components is more intensive than additive noise in the luminance component | Y | 14 | Non eccentricity pattern noise | Y 
3 |  Spatially correlated noise | N | 15 | Local block-wise distortions of different intensity | Y 
4 | Masked noise | N | 16 | Mean shift (intensity shift) | Y 
5 | High frequency noise |  | 17 | Contrast change | Y 
6 | Impulse noise | Y | 18 | Change of color saturation | Y 
7 | Quantization noise | Y | 19 | Multiplicative Gaussian noise | Y 
8 | Gaussian blur | Y | 20 | Comfort noise | N 
9 | Image denoising | Y | 21 | Lossy compression of noisy images | N
10 | JPEG compression | Y | 22 | Image color quantization with dither| Y
11 | JPEG2000 compression | Y | 23 | Chromatic aberrations	| Y 
12 | JPEG transmission errors | N | 24 | Sparse sampling and reconstruction| N


Put your original images in pristine_images folder and run the main function.
```
run tid2013_main.m
```

To generate corrupted images for image inpainting, we use [irregular mask dataset](https://github.com/karfly/qd-imd) and Quick Draw Irregular Mask Dataset , which is combination of 50 million strokes drawn by human hand.

The pristine images from [Cityscapes](https://www.cityscapes-dataset.com/), [COCO-Stuff](https://github.com/nightrome/cocostuff), [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html) and [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ) can be found in their official websites. 

## Pretrained Models

We use pre-trained models to compute the input segmentation or parsing masks, i.e. DeepLabv3 model on Cityscapes dataset, [ResNet50dilated model](https://github.com/CSAILVision/semantic-segmentation-pytorch) for Semantic Segmentation on MIT ADE20K dataset,  [DeepLabV2](https://github.com/kazuto1011/deeplab-pytorch) on COCO-Stuff / PASCAL VOC 2012 dataset, and [BiSeNet](https://github.com/zllrunning/face-parsing.PyTorch) for face parsing on CelebAMask-HQ dataset.


If you found our paper or code useful, please cite our paper.







