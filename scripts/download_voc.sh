# augmented PASCAL VOC
mkdir -p ~/DL_dataset
cd ~/DL_dataset       #save datasets 为$DATASETS
wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz # 1.3 GB
tar -zxvf benchmark.tgz
mv benchmark_RELEASE VOC_aug

# original PASCAL VOC 2012
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar # 2 GB
tar -xvf VOCtrainval_11-May-2012.tar

# mat to png:Data conversions 
# Ground truth labels within augmented PASCAL VOC dataset are distributed as Matlab data files, therefore we will have to convert them before we can start training itself.
cd ~/DL_dataset/VOC_aug/dataset
mkdir cls_png
cd ~/deeplab_v2/voc2012/
./mat2png.py ~/DL_dataset/VOC_aug/dataset/cls ~/DL_dataset/VOC_aug/dataset/cls_png

# label: RGB to 1D
# Those labels in original PASCAL VOC 2012 dataset are defined as RGB images.
mkdir SegmentationClass_1D

cd ~/deeplab_v2/voc2012
./convert_labels.py ~/DL_dataset/VOC2012_orig/SegmentationClass/   ~/DL_dataset/VOC2012_orig/ImageSets/Segmentation/trainval.txt  ~/DL_dataset/VOC2012_orig/SegmentationClass_1D/

# fusion 
cp ~/DL_dataset/VOC2012_orig/SegmentationClass_1D/* ~/DL_dataset/VOC_aug/dataset/cls_png
cp ~/DL_dataset/VOC2012_orig/JPEGImages/* ~/DL_dataset/VOC_aug/dataset/img/

#rename
cd ~/DL_dataset/VOC_aug/dataset
mv ./img ./JPEGImages
mv ./cls_png ./SegmentationClassAug