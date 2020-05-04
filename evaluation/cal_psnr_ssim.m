clear;
im_dir='./evaluation/inpaint/img/';
im_files= dir(fullfile(im_dir,'*.jpg'));

gt_dir='./evaluation/original/img/';
gt_files = dir(fullfile(gt_dir,'*.jpg'));
lengthFiles = length(gt_files);
p=0;s=0;
for i = 1:lengthFiles;
    gt_img = double(imread(strcat(gt_dir,gt_files(i).name)));
    img = double(imread(strcat(im_dir,im_files(i).name)));
    p=p+psnr(img,gt_img)
    s=s+ssim(img,gt_img)
end

ssim=s/lengthFiles;
psnr=p/lengthFiles;
