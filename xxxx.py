#  训练数据生成器
#  batch_size 批大小
#  train_image_path 训练图像路径
#  train_label_path 训练标签路径
#  classNum 类别总数(含背景)
#  colorDict_GRAY 颜色字典
#  resize_shape resize大小
import os
import random
import numpy as np
import cv2
def trainGenerator(batch_size, train_image_path, train_label_path, classNum, colorDict_GRAY, resize_shape=None):
    imageList = os.listdir(train_image_path)
    labelList = os.listdir(train_label_path)
    img = readTif(train_image_path + "\\" + imageList[0])
    #  GDAL读数据是(BandNum,Width,Height)要转换为->(Width,Height,BandNum)
    img = img.swapaxes(1, 0)
    img = img.swapaxes(1, 2)
    #  无限生成数据
    while (True):
        img_generator = np.zeros((batch_size, img.shape[0], img.shape[1], img.shape[2]), np.uint8)
        label_generator = np.zeros((batch_size, img.shape[0], img.shape[1]), np.uint8)
        if (resize_shape != None):
            img_generator = np.zeros((batch_size, resize_shape[0], resize_shape[1], resize_shape[2]), np.uint8)
            label_generator = np.zeros((batch_size, resize_shape[0], resize_shape[1]), np.uint8)
        #  随机生成一个batch的起点
        rand = random.randint(0, len(imageList) - batch_size)
        for j in range(batch_size):
            img = readTif(train_image_path + "\\" + imageList[rand + j])
            img = img.swapaxes(1, 0)
            img = img.swapaxes(1, 2)
            #  改变图像尺寸至特定尺寸(
            #  因为resize用的不多，我就用了OpenCV实现的，这个不支持多波段，需要的话可以用np进行resize
            if (resize_shape != None):
                img = cv2.resize(img, (resize_shape[0], resize_shape[1]))

            img_generator[j] = img

            label = readTif(train_label_path + "\\" + labelList[rand + j]).astype(np.uint8)
            #  若为彩色，转为灰度
            if (len(label.shape) == 3):
                label = label.swapaxes(1, 0)
                label = label.swapaxes(1, 2)
                label = cv2.cvtColor(label, cv2.COLOR_RGB2GRAY)
            if (resize_shape != None):
                label = cv2.resize(label, (resize_shape[0], resize_shape[1]))
            label_generator[j] = label
        img_generator, label_generator = dataPreprocess(img_generator, label_generator, classNum, colorDict_GRAY)
        yield (img_generator, label_generator)