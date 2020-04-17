#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: xiaweihao
"""

import os
from scipy.io import loadmat as sio
import numpy as np
import PIL.Image as Image


os.chdir('/deeplab_v2/voc2012')
path=os.getcwd()
files=os.listdir(path)
labels_path=os.path.join(path,'labels')


palette=[]
for i in range(256):
    palette.extend((i,i,i))
palette[:3*2]=np.array([[0, 0, 0],
                            [128, 0, 0]], dtype='uint8').flatten()



for afile in files:
    file_path=os.path.join(path,afile)
    if os.path.isfile(file_path):
        if os.path.getsize(file_path)==0:
            continue
        mat_idx=afile[:afile.find('.mat')]
        mat_file=sio(file_path)
        mat_file=mat_file['data']
        labels=np.argmax(mat_file,axis=2).astype(np.uint8)
        label_img=Image.fromarray(labels.reshape(labels.shape[0],labels.shape[1]))
        label_img.putpalette(palette)
        label_img=label_img.transpose(Image.FLIP_LEFT_RIGHT)
        label_img = label_img.rotate(90)
        dst_path=os.path.join(labels_path,mat_idx+'.png')
        label_img.save(dst_path)