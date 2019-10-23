#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : JaeZheng
# @Time    : 2019/10/22 21:04
# @File    : test.py

import keras as K
import tensorflow as tf
import cv2


def mean_iou(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    y_pred = K.reshape(y_pred, (-1, nb_classes))
    y_true = tf.to_int32(K.reshape(y_true, (-1, 1))[:,0])
    y_true = K.one_hot(y_true, nb_classes)
    true_pixels = K.argmax(y_true, axis=-1) # exclude background
    pred_pixels = K.argmax(y_pred, axis=-1)
    iou = []
    flag = tf.convert_to_tensor(-1, dtype='float64')
    for i in range(nb_classes-1):
        true_labels = K.equal(true_pixels, i)
        pred_labels = K.equal(pred_pixels, i)
        inter = tf.to_int32(true_labels & pred_labels)
        union = tf.to_int32(true_labels | pred_labels)
        cond = (K.sum(union) > 0) & (K.sum(tf.to_int32(true_labels)) > 0)
        res = tf.cond(cond, lambda: K.sum(inter)/K.sum(union), lambda: flag)
        iou.append(res)
    iou = tf.stack(iou)
    legal_labels = tf.greater(iou, flag)
    iou = tf.gather(iou, indices=tf.where(legal_labels))
    return K.mean(iou)

