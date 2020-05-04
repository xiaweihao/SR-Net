# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
label_list =  ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
            'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
import numpy as np
def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])
def labelcolormap(N):
    if N == 19: # CelebAMask-HQ
        cmap = np.array([(0,  0,  0), (0, 0, 153), (76, 153, 0),
                     (204, 204, 0), (51, 51, 255), (204, 0, 204), (0, 255, 255),
                     (51, 255, 255), (102, 51, 0), (255, 0, 0), (102, 204, 0),
                     (255, 255, 0),(255, 153, 51) , (0, 0, 204), (255, 51, 153), 
                     (0, 204, 204), (0, 51, 0), (204, 0, 0), (0, 204, 0)], 
                     dtype=np.uint8) 
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7-j))
                g = g ^ (np.uint8(str_id[-2]) << (7-j))
                b = b ^ (np.uint8(str_id[-3]) << (7-j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap

def Colorize(gray_image,n):
        cmap = labelcolormap(n)
        size = np.shape(gray_image)
        color_image = np.zeros([3,size[0], size[1]], dtype=np.uint8)

        for label in range(0, len(cmap)):
            mask = (label == gray_image[:,:,0])
            color_image[0][mask] = cmap[label][0]
            color_image[1][mask] = cmap[label][1]
            color_image[2][mask] = cmap[label][2]

        return color_image

#cmap=labelcolormap(150)
#print(cmap)
        
img_dir='./Dataset/CelebA-HQ-qd'
m_dir='./Dataset/CelebA-HQ-qd/seg'
img_list=os.listdir(m_dir)
for i in img_list:
    gray_image=cv2.imread(os.path.join(m_dir,'%s'%i))
    cimg=Colorize(gray_image,19)
    cimg=cimg.transpose(1,2,0)
    cv2.imwrite(os.path.join(img_dir,'cseg/masked_%s'%i),cimg)
        
#img_dir='./Evaluation/rankiqa17_samples/degrade_seg'    
#for trg_dir in ['CA','CCL','CS','GB','GN','IN','QN']:
#    img_list=os.listdir(os.path.join(img_dir,trg_dir))
#    for img in img_list:
#        gray_image=cv2.imread(os.path.join(img_dir,trg_dir,img))
#        cimg=Colorize(gray_image,19)
#        cimg=cimg.transpose(1,2,0)
#        s_dir=os.path.join(img_dir,'cseg',trg_dir)
#        if not os.path.exists(s_dir):
#            os.makedirs(s_dir)
#        s_n=os.path.join(s_dir,'masked_%s'%img)
#        cv2.imwrite(s_n,cimg)