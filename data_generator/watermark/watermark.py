# 先导入所需的包
#import pygame
from PIL import Image, ImageEnhance
import os, random
import pandas as pd
import argparse
import numpy as np


def parse_param():
    ap = argparse.ArgumentParser()
    ap.add_argument("--angle", type=int, default=10, help="rotate angle")
    ap.add_argument("--size", type=int, default=800, help="rotate angle")
#    ap.add_argument('--input_path', type=str, default='G:/SR-Restore/Dataset/ADEChallengeData2016/images/validation/')
    ap.add_argument('--input_path', type=str, default='G:/SR-Restore/Dataset/CelebA-HQ/img/')
    ap.add_argument('--save_path', type=str, default='G:/SR-Restore/Dataset/CelebA-HQ-wm/images/')
#    ap.add_argument('--gt', type=str, default='F:\dataset\wp-image\clean_image/lw-gt/')
    ap.add_argument("--local", type=str, default=True, help="true for local wp, false for non-local")
    ap.add_argument('--wp', type=str, default='G:/SR-Restore/watermark/logo/')
    ap.add_argument('--opacity', type=float, default=0.98, help="image opacity")
    args = vars(ap.parse_args())
    return args


args = parse_param()


def text2png(text,sz):
    pygame.init()  # 初始化
    B = text  # 变量B需要转图片的文字
    text = u"{0}".format(B)  # 引号内引用变量使用字符串格式化
    # 设置字体大小及路径
    font = pygame.font.Font(
        os.path.join("F:\dataset\wp-image\SourceHanSansSC\TencentSans-W7\ot_ttf", "TencentSans-W7.ttf"), sz)
    # 设置位置及颜色
    rtext = font.render(text, True, (204, 204, 204), (255, 255, 255))
    pygame.image.save(rtext, "p.png")
    png = Image.open("p.png")
    return png


def set_opacity(im, opacity):
    """设置透明度"""

    assert opacity >= 0 and opacity < 1
    if im.mode != "RGBA":
        im = im.convert('RGBA')
    else:
        im = im.copy()
    alpha = im.split()[3]
    alpha = ImageEnhance.Brightness(alpha).enhance(opacity)
    im.putalpha(alpha)
    return im


def transparent_back(img,rgb):
    img = img.convert('RGBA')
    L, H = img.size
    color_0 = img.getpixel((L-2, H-2))

    for h in range(H):
        for l in range(L):
            dot = (l, h)
            color_1 = img.getpixel(dot)
            if (color_0[0] - color_1[0]) <= 35 and (color_0[1] - color_1[1]) <= 35 and (
                    color_0[2] - color_1[2]) <= 35 and (color_1[3] == color_0[3]):
                color_1 = color_1[:-1] + (0,)
                img.putpixel(dot, color_1)
            else:
                img.putpixel(dot, (rgb, rgb, rgb, 255))

    return img


def count_coord(img, a, r):
    x_c = img.size[0] / 2
    y_c = img.size[1] / 2
    r = np.pi * (r / 180)
    b_x = (a[0] - x_c) * np.cos(r) - (img.size[1] - a[1] - y_c) * np.sin(r) + x_c
    b_y = (a[0] - x_c) * np.sin(r) + (img.size[1] - a[1] - y_c) * np.cos(r) + y_c
    return (int(b_x), int(img.size[1] - b_y))


def add_logo(im, mark):
    w, h = im.size[0], im.size[1]
    layer = Image.new('RGBA', (int(w), int(h)), (0, 0, 0, 0))
    ratio = mark.size[0]/mark.size[1]
    heng = args['size']
    zong = int(heng/ratio)
    sz = (heng, zong)
    rgb = random.randint(128,255)
    mark = transparent_back(mark,rgb)
    mark = mark.resize(sz)
    mark = set_opacity(mark, opacity = args['opacity'])

    r = args['angle']
    if args['local']:

        x = random.randint(80, w - sz[0] - 80)
        y = random.randint(80, h - sz[1] - 80)
#        x = random.randint(100, w - sz[0] - 100)
#        y = random.randint(100, h - sz[1] - 100)
        layer.paste(mark, (x, y))
        layer = layer.rotate(r)
        out = Image.composite(layer, im, layer)
        return out
    else:

        for i in range(0, im.size[0], sz[0] * 4):
            for j in range(0, im.size[1], sz[1] * 6):
                layer.paste(mark, (i, j))

        for i in range(sz[0] * 2, im.size[0], sz[0] * 4):
            for j in range(sz[1] * 3, im.size[1], sz[1] * 6):
                layer.paste(mark, (i, j))
        layer = layer.rotate(r)
        out = Image.composite(layer, im, layer)
        return out


def batch_add(path,outdir):
    for root, dirs, files in os.walk(path):
        for i in range(0, len(files)):
            image = Image.open(path + files[i])
#            image=image.resize((256, 256),Image.ANTIALIAS)
            print(files[i])
            if args['local']:
                for root, dirs, file in os.walk(args['wp']):
                    j = random.randint(0, len(file) - 1)
                    mark = Image.open(args['wp'] + file[j])
                    img = add_logo(image, mark)
#                    img.save(outdir + 'll-wp%05d.jpg' % i)
                    img.save(outdir + files[i]) 

            else:
                f = open('F:\dataset\wp-image\clean_image/nl-gt/nll-wp%05d.txt' % i, 'a')
                f.truncate()
                for root, dirs, file in os.walk(args['wp']):
                    j = random.randint(0, len(file) - 1)
                    mark = Image.open(args['wp'] + file[j])
                    img = add_logo(image, mark)
                    img.save(outdir + 'nll-wp%05d.jpg' % i)



if __name__ == '__main__':
    path = args['input_path']
    outdir = args['save_path']
    batch_add(path, outdir)

