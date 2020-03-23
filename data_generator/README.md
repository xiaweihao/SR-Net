# Preparing Data

My environment is pytorch0.4 and python3, the code is not tested with other environments, but it should also work on similar environments.

## Get datasets

### Cityscapes
Register and download the dataset from the official [website](https://www.cityscapes-dataset.com/). You can use the [script](SR-Restore/scripts/download_citys.sh) to download the dataset.

You can directly download leftImg8bit_trainvaltest.zip ([11GB](https://www.cityscapes-dataset.com/file-handling/?packageID=3)) and gtFine_trainvaltest.zip ([241MB](https://www.cityscapes-dataset.com/file-handling/?packageID=1)) after registeration.

For leftImg8bit_trainvaltest_foggy.zip ([30GB](https://www.cityscapes-dataset.com/file-handling/?packageID=29)) and leftImg8bit_trainval_rain.zip ([21GB](https://www.cityscapes-dataset.com/file-handling/?packageID=33)), contact [Cityscitys Team](mailto:mail@cityscapes-dataset.net) to get the permission.

### Others

Download from the offical website: 

[CelebaMAsk-HQ](https://github.com/switchablenorms/CelebAMask-HQ), 
[Pascal VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) and [VOCAUG](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz),
[COCO-Stuff](https://github.com/nightrome/cocostuff),
[ADE20K](http://groups.csail.mit.edu/vision/datasets/ADE20K/).

## Data Generation

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

For Gaussian Blur, GaussianNoise, JPEG compression and Chromatic Aberrations, the author of [RankIQA](https://github.com/xialeiliu) released their [code](https://github.com/xialeiliu/RankIQA/tree/master/data/rank_tid2013) to generate ranking dataset for tid2013 dataset (17 distortions).

For watermark, the scripts are at [Repo](https://github.com/xiaweihao/SR-Restore/tree/master/data_generator/watermark).

For Fog, We use FoggyCitysapes on the [Cityscapes](https://www.cityscapes-dataset.com/). If you want to generate on your own, [fog simulation pipeline](https://www.vision.ee.ethz.ch/~csakarid/Model_adaptation_SFSU_dense/) by Christos Sakaridis are available on [fog_simulation_DBF](https://github.com/sakaridis/fog_simulation_DBF).

For Rain, we use RainyCitysapes on the [Cityscapes](https://www.cityscapes-dataset.com/). This is a new dataset published on CVPR 2019 with rain and fog based on the formulation of rain images.

For Reflection, we use a modified [code](https://github.com/xiaweihao/SR-Restore/tree/master/data_generator/reflect) of [ERRNet](https://github.com/Vandermode/ERRNet) to generate reflection.