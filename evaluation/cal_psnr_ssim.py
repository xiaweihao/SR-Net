import skimage
import cv2
import os


def calc_psnr(im1, im2):
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    return compare_psnr(im1_y, im2_y)

def calc_ssim(im1, im2):
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    return compare_ssim(im1_y, im2_y)

def to_grey(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ssim, psnr=0.0,0.0

#------------------------------------------------------------------
#gtim_dir='./evaluation/rankiqa17_samples/level4/img'
#refim_dir='./evaluation/rankiqa17_samples/level4/degrade'
#
#for trg_dir in ['CA','GB','GN','JPEG']:
#    gt_list=os.listdir(os.path.join(gtim_dir))
#    ref_list=os.listdir(os.path.join(refim_dir,trg_dir))
#    

#    
#    from skimage.measure import compare_psnr, compare_ssim
#    for img in gt_list:
#    #    ref = cv2.imread(os.path.join(refim_dir,img.replace('jpg','bmp')),-1)
#        ref = cv2.imread(os.path.join(refim_dir,trg_dir,img),-1)
#        gt = cv2.imread(os.path.join(gtim_dir,img),-1)
##        ssim+=compare_ssim(ref,gt,multichannel=True)
##        psnr+=compare_psnr(ref,gt)
##        
#        compute_psnr = cv2.PSNR(ref, gt)
#        compute_ssim = compare_ssim(to_grey(ref), to_grey(gt))
#        psnr += compute_psnr
#        ssim += compute_ssim
#    n=len(gt_list)
#    psnr = psnr / n
#    ssim = ssim / n
#    print('%s:psnr%f,ssim%f'%(trg_dir,psnr,ssim))
#------------------------------------------------------------------
from skimage.measure import compare_ssim
gtim_dir='./evaluation/original/img'
refim_dir='./evaluation/inpaint_qd/img'
gt_list =os.listdir(os.path.join(gtim_dir))
for img in gt_list:
    ref = cv2.imread(os.path.join(refim_dir,img),-1)
    gt = cv2.imread(os.path.join(gtim_dir,img),-1)
    compute_psnr = cv2.PSNR(ref, gt)
    compute_ssim = compare_ssim(to_grey(ref), to_grey(gt))
    psnr += compute_psnr
    ssim += compute_ssim
n=len(gt_list)
psnr = psnr / n
ssim = ssim / n
print('psnr%f,ssim%f'%(psnr,ssim))


