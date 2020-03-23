from __future__ import division
import math
import random
from PIL import Image, ImageOps, ImageEnhance, PILLOW_VERSION
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import scipy.stats as st
import cv2
import numbers
import types
import random
import collections
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


# utility
def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def arrshow(arr):
    Image.fromarray(arr.astype(np.uint8)).show()


def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    h = math.ceil(h / 2.) * 2  # round up to even
    return img.resize((w, h), Image.BICUBIC)


# functional 
def gaussian_blur(img, kernel_size, sigma):
    from scipy.ndimage.filters import gaussian_filter
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    img = np.asarray(img)
    # the 3rd dimension (i.e. inter-band) would be filtered which is unwanted for our purpose
    # new = gaussian_filter(img, sigma=sigma, truncate=truncate)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    elif isinstance(kernel_size, collections.Sequence):
        assert len(kernel_size) == 2        
    new = cv2.GaussianBlur(img, kernel_size, sigma)  # apply gaussian filter band by band    
    return Image.fromarray(new)


# transforms
class GaussianBlur(object):
    def __init__(self, kernel_size=11, sigma=3):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, img):
        return gaussian_blur(img, self.kernel_size, self.sigma)


class ReflectionSythesis_1(object):
    """Reflection image data synthesis for weakly-supervised learning 
    of ICCV 2017 paper *"A Generic Deep Architecture for Single Image Reflection Removal and Image Smoothing"*    
    """
    def __init__(self, kernel_sizes=None, low_sigma=2, high_sigma=5, low_gamma=1.3, high_gamma=1.3):
        self.kernel_sizes = kernel_sizes or [11]
        self.low_sigma = low_sigma
        self.high_sigma = high_sigma
        self.low_gamma = low_gamma
        self.high_gamma = high_gamma
        print('[i] reflection sythesis model: {}'.format({
            'kernel_sizes': kernel_sizes, 'low_sigma': low_sigma, 'high_sigma': high_sigma,
            'low_gamma': low_gamma, 'high_gamma': high_gamma}))

    def __call__(self, B, R):
        if not _is_pil_image(B):
            raise TypeError('B should be PIL Image. Got {}'.format(type(B)))
        if not _is_pil_image(R):
            raise TypeError('R should be PIL Image. Got {}'.format(type(R)))
        
        B_ = np.asarray(B, np.float32) / 255.
        R_ = np.asarray(R, np.float32) / 255.
        
        kernel_size = np.random.choice(self.kernel_sizes)
        sigma = np.random.uniform(self.low_sigma, self.high_sigma)
        gamma = np.random.uniform(self.low_gamma, self.high_gamma)
        R_blur = R_
        kernel = cv2.getGaussianKernel(11, sigma)
        kernel2d = np.dot(kernel, kernel.T)

        for i in range(3):
            R_blur[...,i] = convolve2d(R_blur[...,i], kernel2d, mode='same')
        
        if np.shape(B_)==np.shape(R_blur):
            M_ = B_ + R_blur
            if np.max(M_) > 1:
                m = M_[M_ > 1]
                m = (np.mean(m) - 1) * gamma
                R_blur = np.clip(R_blur - m, 0, 1)
                M_ = np.clip(R_blur + B_, 0, 1)
        else:   
            R_blur = cv2.resize(R_blur,(2048,1024))
 
            alpha = 0.7
            beta = 1-alpha
            gamma = 0
            M_ = cv2.addWeighted(B_, alpha, R_blur, beta, gamma)
#            M_ = B_ + R_blur
            if np.max(M_) > 1:
                m = M_[M_ > 1]
                m = (np.mean(m) - 1) * gamma
                R_blur = np.clip(R_blur - m, 0, 1)
                M_ = np.clip(R_blur + B_, 0, 1)
                
#            rows, cols = R_blur.shape[:2]
#            M_ = B_
#            roi = B_[:rows, :cols]
#            dst = cv2.add(roi, R_blur)  
#     
#            M_[:rows, :cols] = dst  
#        
#            if np.max(M_) > 1:
#                m = M_[M_ > 1]
#                m = (np.mean(m) - 1) * gamma
#                R_blur = np.clip(R_blur - m, 0, 1)
#                dst = np.clip(R_blur + roi, 0, 1)
#                M_[:rows, :cols] = dst 
        
        return B_, R_blur, M_


class Sobel(object):
    def __call__(self, img):
        if not _is_pil_image(img):
            raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

        gray_img = np.array(img.convert('L'))
        x = cv2.Sobel(gray_img,cv2.CV_16S,1,0)
        y = cv2.Sobel(gray_img,cv2.CV_16S,0,1)
        
        absX = cv2.convertScaleAbs(x)   
        absY = cv2.convertScaleAbs(y)
        
        dst = cv2.addWeighted(absX,0.5,absY,0.5,0)
        return Image.fromarray(dst)


class ReflectionSythesis_2(object):
    """Reflection image data synthesis for weakly-supervised learning 
    of CVPR 2018 paper *"Single Image Reflection Separation with Perceptual Losses"*
    """
    def __init__(self, kernel_sizes=None):
        self.kernel_sizes = kernel_sizes or np.linspace(1,5,80)
    
    @staticmethod
    def gkern(kernlen=100, nsig=1):
        """Returns a 2D Gaussian kernel array."""
        interval = (2*nsig+1.)/(kernlen)
        x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw/kernel_raw.sum()
        kernel = kernel/kernel.max()
        return kernel

    def __call__(self, t, r):        
        t = np.float32(t) / 255.
        r = np.float32(r) / 255.
        ori_t = t
        # create a vignetting mask
        g_mask=self.gkern(560,3)
        g_mask=np.dstack((g_mask,g_mask,g_mask))
        sigma=self.kernel_sizes[np.random.randint(0, len(self.kernel_sizes))]

        t=np.power(t,2.2)
        r=np.power(r,2.2)
        
        sz=int(2*np.ceil(2*sigma)+1)
        
        r_blur=cv2.GaussianBlur(r,(sz,sz),sigma,sigma,0)
        
        if np.shape(r_blur)!=np.shape(t):
            r_blur = cv2.resize(r_blur,(2048,1024))
            
        blend=r_blur+t
        
        att=1.08+np.random.random()/10.0
        
        for i in range(3):
            maski=blend[:,:,i]>1
            mean_i=max(1.,np.sum(blend[:,:,i]*maski)/(maski.sum()+1e-6))
            r_blur[:,:,i]=r_blur[:,:,i]-(mean_i-1)*att
        r_blur[r_blur>=1]=1
        r_blur[r_blur<=0]=0

        h,w=r_blur.shape[0:2]
        neww=np.random.randint(0, 560-w-10)
        newh=np.random.randint(0, 560-h-10)
        alpha1=g_mask[newh:newh+h,neww:neww+w,:]
        alpha2 = 1-np.random.random()/5.0
        r_blur_mask=np.multiply(r_blur,alpha1)
        blend=r_blur_mask+t*alpha2
        
        t=np.power(t,1/2.2)
        r_blur_mask=np.power(r_blur_mask,1/2.2)
        blend=np.power(blend,1/2.2)
        blend[blend>=1]=1
        blend[blend<=0]=0
        
        return np.float32(ori_t), np.float32(r_blur_mask), np.float32(blend)


# Examples
if __name__ == '__main__':
    """cv2 imread"""
    # img = cv2.imread('testdata_reflection_real/19-input.png')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img2 = cv2.GaussianBlur(img, (11,11), 3)    

    """Sobel Operator"""
    # img = np.array(Image.open('datasets/VOC224/train/B/2007_000250.png').convert('L'))


    """Reflection Sythesis"""
#    r = Image.open('CEILNet/testdata_reflection_synthetic/2007_008567-label1.png')
#    b = Image.open('CEILNet/testdata_reflection_synthetic/2008_000287-label1.png')
    bench_dir = 'L:/Datasets/Cityscapes/leftImg8bit/train/'
    reflect_dir = 'L:/Datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/'
    f_n = os.listdir(reflect_dir)
    n = len(f_n)
    for root, dirs, files in os.walk(bench_dir):
        for i in range(0, len(files)):
            b = Image.open(os.path.join(root, files[i]))
            r = Image.open(os.path.join(reflect_dir, f_n[random.randint(0, n)]))
            G = ReflectionSythesis_2()
            b_, r_, m = G(b, r)
            m= cv2.cvtColor(m, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join('./results', files[i]), m*255.0)
