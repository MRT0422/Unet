# -*- encoding: utf-8 -*-
# Author  : Haitong
# Time    : 2020/9/9 15:37
# File    : evalution_segmentaion.py
# Software: PyCharm
from __future__ import division
import numpy as np
import six
import cfg
def calc_semantic_segmentation_confusion(pred_labels, gt_labels):
    pred_labels = iter(pred_labels)
    gt_labels = iter(gt_labels)     #
    n_class = cfg.DATASET[1]
    confusion = np.zeros((n_class, n_class), dtype=np.int64)
    for pred_label, gt_label in six.moves.zip(pred_labels, gt_labels):  #six.moves.zip（）
        if pred_label.ndim != 2 or gt_label.ndim != 2:          #获取pred_label的维数
            raise ValueError('ndim of labels should be two.')
        if pred_label.shape != gt_label.shape:
            raise ValueError('Shape of ground truth and prediction should'
                             ' be same.')
        pred_label = pred_label.flatten()   # (168960, )  #pred_label.flatten 变为一维列表
        gt_label = gt_label.flatten()   # (168960, )

        # Dynamically expand the confusion matrix if necessary.
        lb_max = np.max((pred_label, gt_label))
        # print(lb_max)
        if lb_max >= n_class:          #n_class是否包含了背景类？
            expanded_confusion = np.zeros(
                (lb_max + 1, lb_max + 1), dtype=np.int64)
            expanded_confusion[0:n_class, 0:n_class] = confusion      # 选取 n_class*n_class的矩阵
            n_class = lb_max + 1            #为什么+1
            confusion = expanded_confusion  #获得一个包含所有标签的0矩阵

        # Count statistics from valid pixels.  极度巧妙 × class_nums 正好使得每个ij能够对应.
        mask = gt_label >= 0
        confusion += np.bincount(                                            # np.bincount统计某一个数出现的频次，索引为统计的数值。
            n_class * gt_label[mask].astype(int) + pred_label[mask], minlength = n_class ** 2)\
            .reshape((n_class, n_class))                                     # gt_label[mask].astype(int) + pred_label[mask]哪来的mask属性
         #confusion 结果为混淆矩阵
    for iter_ in (pred_labels, gt_labels):
        # This code assumes any iterator does not contain None as its items.
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same')

    return confusion
def calc_semantic_segmentation_iou(confusion):

    iou_denominator = (confusion.sum(axis=1) + confusion.sum(axis=0)
                       - np.diag(confusion))
    iou = np.diag(confusion) / (iou_denominator + 1e-10)
    return iou[:]
    # return iou

def eval_semantic_segmentation(pred_labels, gt_labels):   #gt_labels真实标签

    confusion = calc_semantic_segmentation_confusion(
        pred_labels, gt_labels)     #confusion混淆矩阵
    iou = calc_semantic_segmentation_iou(confusion)     # (1２, )
    pixel_accuracy = np.diag(confusion).sum() / confusion.sum()
    class_accuracy = np.diag(confusion) / (np.sum(confusion, axis=1) + 1e-10)  # (1２, )

    return {'iou': iou, 'miou': np.nanmean(iou),
            'pixel_accuracy': pixel_accuracy,
            'class_accuracy': class_accuracy,
            'mean_class_accuracy': np.nanmean(class_accuracy)}
            # 'mean_class_accuracy': np.nanmean(class_accuracy)}