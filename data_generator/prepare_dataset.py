import os
import glob
from PIL import Image

def load_resized_img(path):
    return Image.open(path).convert('RGB').resize((256, 256))

def check_matching_pair(segmap_path, photo_path):
    segmap_identifier = os.path.basename(segmap_path).replace('seg', '')
    photo_identifier = os.path.basename(photo_path).replace('img', '')
        
    assert segmap_identifier == photo_identifier, \
        "[%s] and [%s] don't seem to be matching. Aborting." % (segmap_path, photo_path)
    
def process_dataset(seg_dir, img_dir, output_dir, phase):
    save_phase = 'test' if phase == 'val' else 'train'
    savedir = os.path.join(output_dir, save_phase)
    os.makedirs(savedir, exist_ok=True)
    os.makedirs(savedir + '/seg', exist_ok=True)
    os.makedirs(savedir + '/img', exist_ok=True)
    print("Directory structure prepared at %s" % output_dir)
    
    segmap_expr = os.path.join(seg_dir, phase) + "/seg/*.png"
    segmap_paths = glob.glob(segmap_expr)
    segmap_paths = sorted(segmap_paths)

    photo_expr = os.path.join(img_dir, phase) + "/img/*.png"
    photo_paths = glob.glob(photo_expr)
    photo_paths = sorted(photo_paths)

    assert len(segmap_paths) == len(photo_paths), \
        "%d images that match [%s], and %d images that match [%s]. Aborting." % (len(segmap_paths), segmap_expr, len(photo_paths), photo_expr)

    for i, (segmap_path, photo_path) in enumerate(zip(segmap_paths, photo_paths)):
        check_matching_pair(segmap_path, photo_path)
        segmap = load_resized_img(segmap_path)
        photo = load_resized_img(photo_path)

        # data for pix2pix where the two images are placed side-by-side
        seg_new = Image.new('RGB', (256, 256))
        seg_new.paste(segmap, (0, 0))
        seg_savepath = os.path.join(savedir + '/seg', "%d.jpg" % i)
        seg_new.save(seg_savepath, format='JPEG', subsampling=0, quality=100)

        img_new = Image.new('RGB', (256, 256))
        img_new.paste(segmap, (0, 0))
        img_savepath = os.path.join(savedir + '/seg', "%d.jpg" % i)
        img_new.save(img_savepath, format='JPEG', subsampling=0, quality=100)

        if i % (len(segmap_paths) // 10) == 0:
            print("%d / %d: last image saved at %s, " % (i, len(segmap_paths), savepath))

if __name__ == '__main__':

    dataset = 'cityscapes'#'celeba', 'coco'

    process_dataset(dataset + '/data/seg', dataset + '/data/img', '../srnet/' + dataset, 'train')
    print('Done')