#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : JaeZheng
# @Time    : 2019/11/10 8:57
# @File    : preprocess.py

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def crop_patch():
    # 训练集路径
    # image_path = 'data/thyroid/train/image/'
    # label_path = 'data/thyroid/train/label/'
    # 测试集路径
    image_path = 'F:/ThryoidSeg/image/test_image/'
    label_path = 'F:/ThryoidSeg/image/test_label/'
    image_list = os.listdir(image_path)
    # label_list = os.listdir(label_path)
    count = 0
    for image_name in image_list:
        img = image_path + image_name
        mask = label_path + image_name
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
        dx = 64
        dy = 64
        x1 = 0
        y1 = 0
        while y1 < 400:
            while x1 < 500:
                patch_img = img[x1:x1+dx, y1:y1+dy]
                patch_mask = mask[x1:x1+dx, y1:y1+dy]
                if np.mean(patch_mask) != 0:
                    # 训练集patch存储路径
                    # cv2.imwrite('data/thyroid/train_patch/image/'+str(count)+'.bmp', patch_img)
                    # cv2.imwrite('data/thyroid/train_patch/label/'+str(count)+'.bmp', patch_mask)
                    # 测试集patch存储路径
                    cv2.imwrite('data/thyroid/test_patch/' + str(count) + '.bmp', patch_img)
                    cv2.imwrite('data/thyroid/test_patch/' + str(count) + '_true.bmp', patch_mask)
                    count += 1
                x1 += dx
            x1 = 0
            y1 += dy


def resize_picture():
    # 训练集路径
    # image_path = 'data/thyroid/train/image/'
    # label_path = 'data/thyroid/train/label/'
    # 测试集路径
    image_path = 'F:/ThryoidSeg/image/test_image/'
    label_path = 'F:/ThryoidSeg/image/test_label/'
    image_list = os.listdir(image_path)
    # label_list = os.listdir(label_path)
    for image_name in image_list:
        img = image_path + image_name
        mask = label_path + image_name
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
        resize_img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        resize_mask = cv2.resize(mask, (64, 64), interpolation=cv2.INTER_AREA)
        # 训练集存储路径
        # cv2.imwrite('data/thyroid/train_resize/image/'+image_name, resize_img)
        # cv2.imwrite('data/thyroid/train_resize/label/'+image_name, resize_mask)
        # 测试集存储路径
        cv2.imwrite('data/thyroid/test_resize/' + image_name, resize_img)
        cv2.imwrite('data/thyroid/test_resize/' + image_name.split('.')[0] + '_true.bmp', resize_mask)


if __name__ == '__main__':
    # crop_patch()
    resize_picture()