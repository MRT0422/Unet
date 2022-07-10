# -*- encoding: utf-8 -*-
# Author  : Haitong
# Time    : 2021/8/24 10:41
# File    : visualization.py
# Software: PyCharm

import pandas as pd
import numpy as np
import torch as t
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from dataset import LoadDataset
import Unet_m2
import cfg
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
base_dir = os.path.abspath(os.path.dirname(__file__))

def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

def main():
    VISUA_ROOT = cfg.VIS_ROOT
    VISUA_LABEL = cfg.VIS_LABEL
    device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
    Load_test = LoadDataset([VISUA_ROOT, VISUA_LABEL], cfg.crop_size)
    test_data = DataLoader(Load_test, batch_size=1, shuffle=False, num_workers=0)
    net = Unet_m2.UNet(4, cfg.DATASET[1]).to(device)
    weight_path = os.path.abspath(os.path.join(base_dir, "Results", "Unet", "Unettest", "valid_loss.pth"))
    net.load_state_dict(t.load(weight_path))
    net.eval()
    pd_label_color = pd.read_csv(cfg.class_dict_path, sep=',')
    name_value = pd_label_color['name'].values
    num_class = len(name_value)
    colormap = []

    for i in range(num_class):
        tmp = pd_label_color.iloc[i]
        color = [tmp['r'], tmp['g'], tmp['b']]
        colormap.append(color)
    cm = np.array(colormap).astype('uint8')
    dir = "./Results/visualization/"
    makedir(dir)

    for i, sample in enumerate(test_data):
        # 载入
        valImg = sample['img'].to(device)
        # valLabel = sample['label'].long().to(device)
        out = net(valImg)
        out = F.log_softmax(out, dim=1)
        pre_label = out.max(1)[1].squeeze().cpu().data.numpy()
        pre = cm[pre_label]
        pre1 = Image.fromarray(pre)
        pre1.save(dir + str(i) + ".png")
        print('Done' + str(i))

if __name__ == "__main__":
    main()
