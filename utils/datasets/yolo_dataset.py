import os
import cv2
import glob
import random
import numpy as np
from pathlib import Path

import torch
from torch.utils import data
from torch.utils.data import Dataset

from utils.constant import IMG_FORMATS


def contrast_and_brightness(img):
    alpha = random.uniform(0.25, 1.75)
    beta = random.uniform(0.25, 1.75)
    blank = np.zeros(img.shape, img.dtype)
    # dst = alpha * img + beta * blank
    dst = cv2.addWeighted(img, alpha, blank, 1 - alpha, beta)
    return dst


def motion_blur(image):
    if random.randint(1, 2) == 1:
        degree = random.randint(2, 3)
        angle = random.uniform(-360, 360)
        image = np.array(image)

        # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
        M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

        motion_blur_kernel = motion_blur_kernel / degree
        blurred = cv2.filter2D(image, -1, motion_blur_kernel)

        # convert to uint8
        cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
        blurred = np.array(blurred, dtype=np.uint8)
        return blurred
    else:
        return image


def augment_hsv(img, hgain=0.0138, sgain=0.678, vgain=0.36):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)  # no return needed
    return img


def random_resize(img):
    h, w, _ = img.shape
    rw = int(w * random.uniform(0.8, 1))
    rh = int(h * random.uniform(0.8, 1))

    img = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    return img


def img_aug(img):
    img = contrast_and_brightness(img)
    #img = motion_blur(img)
    #img = random_resize(img)
    #img = augment_hsv(img)
    return img


def collate_fn(batch):
    img, label = zip(*batch)
    for i, l in enumerate(label):
        if l.shape[0] > 0:
            l[:, 0] = i
    return torch.stack(img), torch.cat(label, 0)


def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]


class YoloDataset():

    def __init__(self, path, img_size=(640, 640), imgaug=False, prefix=''):
        assert os.path.exists(path), f"{path}文件路径错误或不存在"

        self.path = path
        self.img_size_width, self.img_size_height = img_size
        self.imgaug = imgaug

        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise Exception(f'{p} does not exist')
            self.im_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert self.im_files, f'No images found'
        except Exception as e:
            raise Exception(f'Error loading data from {path}: {e}')
        
        #  Check cache
        self.label_files = img2label_paths(self.im_files)  # labels

    def __getitem__(self, index):
        img_path = self.im_files[index]
        label_path = self.label_files[index]

        # 归一化操作
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.img_size_width, self.img_size_height), interpolation=cv2.INTER_LINEAR)
        #数据增强
        if self.imgaug == True:
            img = img_aug(img)
        img = img.transpose(2, 0, 1)

        # 加载label文件
        if os.path.exists(label_path):
            label = []
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    l = line.strip().split("\t")
                    label.append([0, l[0], l[1], l[2], l[3], l[4]])
            label = np.array(label, dtype=np.float32)

            if label.shape[0]:
                assert label.shape[1] == 6, '> 5 label columns: %s' % label_path
                #assert (label >= 0).all(), 'negative labels: %s'%label_path
                #assert (label[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels: %s'%label_path
        else:
            raise Exception("%s is not exist" % label_path)

        return torch.from_numpy(img), torch.from_numpy(label)

    def __len__(self):
        return len(self.data_list)


if __name__ == "__main__":
    data = YoloDataset("/home/xuehao/Desktop/TMP/pytorch-yolo/widerface/train.txt")
    img, label = data.__getitem__(0)
    print(img.shape)
    print(label.shape)
