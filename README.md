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

## Evaluation Methods and Results

### DeRain

Comparison with the state-of-the-arts in terms of the PSNR and SSIM on the test set of RainCityscapes.     

No. | Method | PSNR | SSIM 
------ | ------ | ------ | ------ 
1 | [DSC](http://www.math.nus.edu.sg/~matjh/download/image_deraining/rain_removal_v.1.1.zip) 	| 16.25 | 0.7746
2 | [GMMLP](http://yu-li.github.io/)	| 17.80 | 0.8169
3 | [JOB](http://openaccess.thecvf.com/content_iccv_2017/html/Zhu_Joint_Bi-Layer_Optimization_ICCV_2017_paper.html)	| 15.10 | 0.7592
4 | [RESCAN](https://xialipku.github.io/RESCAN/)	| 24.49 | 0.8852
5 | [DID-MDN](https://github.com/hezhangsprinter/DID-MDN)	| 28.43 | 0.9349
6 | [SPANet](https://github.com/stevewongv/SPANet)	| 25.39 | 0.8933
7 | [PReNet](https://github.com/YadiraF/PRNet)	| 25.96 | 0.9147
8 | [DAF-Net](https://github.com/xw-hu/DAF-Net)	| 30.06 | 0.9530 
9 | Ours	| 32.41 | 0.9579
 
### DeFog

Comparison with the state-of-the-arts in terms of the PSNR and SSIM on the test set of FoggyCityscapes.

No. | Method | PSNR | SSIM 
------ | ------ | ------ | ------ 
1 | [DCP](https://github.com/raven-dehaze-work/DCP-Dehaze) 	| 23.98 | 0.8349
2 | [NLID](https://github.com/qub-blesson/DeFog) 	| 24.43 | 0.8512
3 | [MSCNN](https://github.com/raven-dehaze-work/MSCNN_MATLAB) 	| 29.36 | 0.9317 
4 | Ours 		| 32.64 | 0.9618

### DeReflect

Comparison with the state-of-the-arts in terms of the PSNR and SSIM on the test set of ReflectCityscapes.

No. | Method | PSNR | SSIM 
------ | ------ | ------ | ------ 
1 | [LB14]() | 29.23 | 0.9337 
2 | [CEILNet](https://github.com/fqnchina/CEILNet) | 24.51 | 0.8826
3 | [PLRS](http://yu-li.github.io/paper/li_cvpr14_layer.pdf) | 28.06 | 0.9182 
4 | [BDN](https://github.com/yangj1e/bdn-refremv) | 15.10 | 0.7592 
5 | [ERRNet](https://github.com/Vandermode/ERRNet) | 30.80 | 0.9369
6 | Ours | 32.06 | 0.9590

### Others

We conduct experiments on various datasets, which mostly can be found in our [paper](http://arxiv.org/abs/1911.00679). Below are semantic segmentation and image restoration results of watermark and iirgular mask.

<p align="center">
  <img src="/asserts/celeba_wm_results.jpg">
</p>

## Data Generation

If you found our paper or code useful, please cite our paper.
```
@article{xia2019adverse,
  author    = {Weihao Xia and Zhanglin Cheng and Yujiu Yang and Jing-Hao Xue},
  title     = {Cooperative Semantic Segmentation and Image Restoration in Adverse Environmental Conditions},
  year      = {2019},
  url       = {http://arxiv.org/abs/1911.00679},
}
```







