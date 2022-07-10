#  数据预处理：图像归一化+标签onehot编码
#  img 图像数据
#  label 标签数据
#  classNum 类别总数(含背景)
#  colorDict_GRAY 颜色字典
import numpy as np
def dataPreprocess(img, label, classNum, colorDict_GRAY):
    #  归一化
    img = img / 255.0
    for i in range(colorDict_GRAY.shape[0]):
        label[label == colorDict_GRAY[i][0]] = i
    #  将数据厚度扩展到classNum(包括背景)层
    new_label = np.zeros(label.shape + (classNum,))
    #  将平面的label的每类，都单独变成一层
    for i in range(classNum):
        new_label[label == i,i] = 1
    label = new_label
    return (img, label)