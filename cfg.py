# -*- encoding: utf-8 -*-
# Author  : Haitong
# Time    : 2021/6/25 21:45
# File    : cfg.py
# Software: PyCharm


DATASET = ['GID',3]
BATCH_SIZE = 2
EPOCH_NUMBER =5

crop_size = (512, 512)

class_dict_path = 'E:/Deep Learning/Deep Learnimg data/Deep Learning data/class_dict.csv'      #地类RGB颜色表

TRAIN_ROOT = 'E:/Deep Learning/Deep Learnimg data/Deep Learning data/train-img'                #训练数据路径
TRAIN_LABEL = 'E:/Deep Learning/Deep Learnimg data/Deep Learning data/train-test'


VAL_ROOT = 'E:/Deep Learning/Deep Learnimg data/Deep Learning data/val-img'                    #验证数据路径
VAL_LABEL = 'E:/Deep Learning/Deep Learnimg data/Deep Learning data/val-test'


TEST_ROOT = 'E:/Deep Learning/Deep Learnimg data/Deep Learning data/test-img'                  #测试数据路径
TEST_LABEL = 'E:/Deep Learning/Deep Learnimg data/Deep Learning data/test-test'


