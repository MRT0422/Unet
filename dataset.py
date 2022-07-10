# -*- encoding: utf-8 -*-
# Author  : Haitong
# Time    : 2020/9/8 16:21
# File    : dataset.py
# Software: PyCharm

import pandas as pd
import os
import torch as t
import numpy as np
import torchvision.transforms.functional as ff
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import cfg
from osgeo import gdal
import cv2

class LabelProcessor:
    """对标签图像的编码"""

    def color_dict(labelFolder, classNum):
        colorDict = []
        #  获取文件夹内的文件名
        ImageNameList = os.listdir(labelFolder)
        for i in range(len(ImageNameList)):
            ImagePath = labelFolder + "/" + ImageNameList[i]
            img = cv2.imread(ImagePath).astype(np.uint32)
            #  如果是灰度，转成RGB
            if (len(img.shape) == 2):
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.uint32)
            #  为了提取唯一值，将RGB转成一个数
            img_new = img[:, :, 0] * 1000000 + img[:, :, 1] * 1000 + img[:, :, 2]
            unique = np.unique(img_new)
            #  将第i个像素矩阵的唯一值添加到colorDict中
            for j in range(unique.shape[0]):
                colorDict.append(unique[j])
            #  对目前i个像素矩阵里的唯一值再取唯一值
            colorDict = sorted(set(colorDict))
            #  若唯一值数目等于总类数(包括背景)ClassNum，停止遍历剩余的图像
            if (len(colorDict) == classNum):
                break
        #  存储颜色的RGB字典，用于预测时的渲染结果
        colorDict_RGB = []
        for k in range(len(colorDict)):
            #  对没有达到九位数字的结果进行左边补零(eg:5,201,111->005,201,111)
            color = str(colorDict[k]).rjust(9, '0')
            #  前3位R,中3位G,后3位B
            color_RGB = [int(color[0: 3]), int(color[3: 6]), int(color[6: 9])]
            colorDict_RGB.append(color_RGB)
        #  转为numpy格式
        colorDict_RGB = np.array(colorDict_RGB)
        #  存储颜色的GRAY字典，用于预处理时的onehot编码
        colorDict_GRAY = colorDict_RGB.reshape((colorDict_RGB.shape[0], 1, colorDict_RGB.shape[1])).astype(np.uint8)
        colorDict_GRAY = cv2.cvtColor(colorDict_GRAY, cv2.COLOR_BGR2GRAY)
        return colorDict_RGB, colorDict_GRAY
class LoadDataset(Dataset):
    def __init__(self, file_path=[], crop_size=None):
        """para:
            file_path(list): 数据和标签路径,列表元素第一个为图片路径，第二个为标签路径
        """
        # 1 正确读入图片和标签路径
        if len(file_path) != 2:
            raise ValueError("同时需要图片和标签文件夹的路径，图片路径在前")
        self.img_path = file_path[0]
        self.label_path = file_path[1]
        # 2 从路径中取出图片和标签数据的文件名保持到两个列表当中（程序中的数据来源）
        self.imgs = self.read_file(self.img_path)
        self.labels = self.read_file(self.label_path)
        # 3 初始化数据处理函数设置

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        img = gdal.Open(img)
        label = Image.open(label)
        image = np.zeros((img.RasterYSize, img.RasterXSize, img.RasterCount), dtype='float')
        for b in range(img.RasterCount):
            band = img.GetRasterBand(b+1)
            image[:, :, b] =band.ReadAsArray()
        sample = {'img': img, 'label': label}
        return sample

    def __len__(self):
        return len(self.imgs)

    def read_file(self, path):#返回每个文件的具体路径
        """从文件夹中读取数据"""
        dataset = gdal.Open(path)
        width = dataset.RasterXSize
        height = dataset.RasterYSize
        GdalImg_data = dataset.ReadAsArray(0, 0, width, height)
        return GdalImg_data

label_processor = LabelProcessor(cfg.class_dict_path)

if __name__ == "__main__":
    train = LoadDataset([cfg.TRAIN_ROOT, cfg.TRAIN_LABEL], cfg.crop_size)
    val = LoadDataset([cfg.VAL_ROOT, cfg.VAL_LABEL], cfg.crop_size)
    if cfg.TEST_ROOT is not None:
        test = LoadDataset([cfg.TEST_ROOT, cfg.TEST_LABEL], cfg.crop_size)
