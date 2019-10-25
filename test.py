#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : JaeZheng
# @Time    : 2019/10/22 21:04
# @File    : test.py

import cv2
import numpy as np


def iou(y_true, y_pred):
    TP, FP, FN = 0, 0, 0
    h, w = y_pred.shape
    for row in range(h):
        for col in range(w):
            cur_pixel = y_pred[row][col]
            if cur_pixel == 255:
                if y_true[row][col] == cur_pixel:
                    TP += 1
                else:
                    FP += 1
            elif cur_pixel == 0 and y_true[row][col] == 255:
                FN += 1
    return TP/(TP+FP+FN)


def iou_2(y_true, y_pred):
    y_true_mask = (y_true == 255)
    y_pred_mask = (y_pred == 255)
    iou = np.sum(y_true_mask & y_pred_mask) / np.sum(y_true_mask | y_pred_mask)
    return iou

mean_iou = []
for i in range(159):
    y_true = cv2.imread("./data/thyroid/test_true/"+str(i)+".bmp", cv2.IMREAD_GRAYSCALE)
    y_pred = cv2.imread("./data/thyroid/test/"+str(i)+"_predict.png", cv2.IMREAD_GRAYSCALE)
    y_pred[y_pred[:,:]<127] = 0
    y_pred[y_pred[:,:]>=127] = 255
    mean_iou.append(iou(y_true, y_pred))
print(np.mean(mean_iou))