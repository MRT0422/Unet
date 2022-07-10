# -*- encoding: utf-8 -*-
# Author  : Haitong
# Time    : 2021/6/25 21:48
# File    : train_Unet.py
# Software: PyCharm
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from dataset import LoadDataset
from evalution_segmentaion import eval_semantic_segmentation
import Unet_m2
import cfg
import pandas  as pd
import time
import os

# import progressbar
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def makedir(new_dir):
    if not os.path.exists(new_dir):     # os.path.exists()就是判断括号里的文件是否存在的意思，括号内的可以是文件路径。
        os.makedirs(new_dir)

if __name__ == "__main__":
    dir1 = "Unet"
    dir2 = dir1 + "test"

    base_dir = os.path.abspath(os.path.dirname(__file__) + os.path.sep + "Results" + os.path.sep + dir1 + os.path.sep + dir2)
    makedir(base_dir)
    #os.path.sep=路径分隔符

    num_class = cfg.DATASET[1]
    # device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
    device = 'cpu'
    Load_train = LoadDataset([cfg.TRAIN_ROOT, cfg.TRAIN_LABEL], cfg.crop_size)
    train_data = DataLoader(Load_train, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)

    Load_val = LoadDataset([cfg.VAL_ROOT, cfg.VAL_LABEL], cfg.crop_size)
    val_data = DataLoader(Load_val, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)

    Load_test = LoadDataset([cfg.TEST_ROOT, cfg.TEST_LABEL], cfg.crop_size)
    test_data = DataLoader(Load_test, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)

    net = Unet_m2.UNet(4,num_class)   #波段数，类别数

    net.to(device)

    criterion = nn.NLLLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    elopad_time = 0
    train_best_miou = [0]
    train_best_loss = [float("inf")]

    val_best_miou = [0]
    val_best_loss = [float("inf")]
    # print(cfg.EPOCH_NUMBER)
    for epoch in range(cfg.EPOCH_NUMBER):
        epoch_start_time = time.time()
        print('Epoch is [{}/{}]'.format(epoch + 1, cfg.EPOCH_NUMBER))
        if epoch % 50 == 0 and epoch != 0:
            for group in optimizer.param_groups:
                group['lr'] *= 0.5

        net.train()
        train_loss = 0
        train_miou = 0
        for i, sample in enumerate(train_data):#报错点
            img_data = sample['img'].to(device)
            img_label = sample['label'].to(device)
            out = net(img_data)               #网络预测结果
            out = F.log_softmax(out, dim=1)   # F.log_softmax并无降维的作用，对每一行进行log_softmax运算，每一行的和等于1（概率）
            loss = criterion(out, img_label)  #计算损失
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            # 评估
            pre_label = out.max(dim=1)[1].data.cpu().numpy() # pre_label网络输出结果
            pre_label = [i for i in pre_label] # 预测的结果
            true_label = img_label.data.cpu().numpy()
            true_label = [i for i in true_label] # 真正的标签
            eval_metrix = eval_semantic_segmentation(pre_label, true_label)
            train_miou += eval_metrix['miou']

        if max(train_best_miou) <= train_miou / len(train_data):
            train_best_miou.append(train_miou / len(train_data))
            t.save(net.state_dict(), os.path.join(base_dir, "train_miou.pth"))           # 2021年6月26日09:13:06

        if min(train_best_loss) >= train_loss / len(train_data):
            train_best_loss.append(train_loss / len(train_data))
            t.save(net.state_dict(), os.path.join(base_dir, "train_loss.pth"))           # 2021年6月26日09:13:06

        train_str = '|Train Loss|: {:.5f}\t|Train MIoU|: {:.5f}\t|Train Best MIoU|: {:.5f}'.format(
            train_loss / len(train_data),
            train_miou / len(train_data),
            max(train_best_miou),
        )
        train_elapsed_time = time.time() - epoch_start_time
        print(train_str)
        # valid
        #以上为一个EPOCH内训练集的参数误差
        time_eval_start = time.time()
        net.eval()
        eval_loss = 0
        eval_miou = 0

        with t.no_grad():
            for j, sample in enumerate(val_data):
                valImg = sample['img'].to(device)
                valLabel = sample['label'].long().to(device)
                out = net(valImg)
                out = F.log_softmax(out, dim=1)
                loss = criterion(out, valLabel)
                eval_loss = loss.item() + eval_loss
                pre_label = out.max(dim=1)[1].data.cpu().numpy()
                pre_label = [i for i in pre_label]
                true_label = valLabel.data.cpu().numpy()
                true_label = [i for i in true_label]
                eval_metrics = eval_semantic_segmentation(pre_label, true_label)
                eval_miou += eval_metrics['miou']

            if max(val_best_miou) <= eval_miou / len(val_data):
                val_best_miou.append(eval_miou / len(val_data))
                t.save(net.state_dict(), os.path.join(base_dir, "valid_miou.pth"))          # 2021年6月26日09:13:06

            if min(val_best_loss) >= eval_loss / len(val_data):
                val_best_loss.append(eval_loss / len(val_data))
                t.save(net.state_dict(), os.path.join(base_dir, "valid_loss.pth"))          # 2021年6月26日09:13:06

            val_str = '|Valid Loss|: {:.5f}\t|Valid MIoU|: {:.5f}\t|Valid Best MIoU|: {:.5f}'.format(
                eval_loss / len(val_data),
                eval_miou / len(val_data),
                max(val_best_miou),
            )
            eval_epoch_time = time.time() - time_eval_start

            print(val_str)
            print("The epoch of {}\ttrain_elapsed_time: {}s\teval_elapsed_time: {}s".format(epoch + 1, round(train_elapsed_time, 1),round(eval_epoch_time, 1)))

        if (epoch + 1) % 1 == 0:
            name1 = ["train_best_miou", "val_best_miou"]
            list1 = [train_best_miou, val_best_miou]
            loss1 = pd.DataFrame(index=name1, data=list1)
            loss1.to_csv(os.path.join(base_dir, "train_val_epoch.csv"))       # 2021年6月26日09:20:59

        elopad_time += time.time() - epoch_start_time
        print("Already elapsed_time: {} min or {}h".format(round(elopad_time / 60, 2), round(elopad_time / 3600, 3)))


    # # -----------------------------test

    # 2021年6月26日10:57:46
    dice_list = []
    acc_list = []
    miou_list = []
    mpa_list = []
    class_acc_list = []

    tt = 0
    dic = ['train_loss', "train_miou", 'valid_loss', 'valid_miou']
    for ii in range(len(dic)):
        train_acc = 0
        train_miou = 0
        train_class_acc = 0
        train_mpa = 0
        time_test_start = time.time()
        net.eval()
        net.to(device)
        time1 = time.time()
        net.load_state_dict(t.load(os.path.join(base_dir, dic[ii]+'.pth')))        # 2021年6月26日09:22:34

        with t.no_grad():
            for i, sample in enumerate(test_data):
                data = sample['img'].to(device)
                label = sample['label'].to(device)
                out = net(data)   # （B C H W）
                out = F.log_softmax(out, dim=1)  #（0.2,0.3 0.5）  # （B C H W）
                pre_label = out.max(dim=1)[1].data.cpu().numpy()   # （B H W）
                pre_label = [i for i in pre_label]
                true_label = label.data.cpu().numpy()
                true_label = [i for i in true_label]
                eval_metrix = eval_semantic_segmentation(pre_label, true_label)

                train_acc = eval_metrix['mean_class_accuracy']
                train_miou = eval_metrix['miou'] + train_miou
                train_mpa = eval_metrix['pixel_accuracy'] + train_mpa
                train_class_acc = train_class_acc + eval_metrix['class_accuracy']

            epoch_str = ('test_miou:{:.5f}\ttest_acc :{:.5f}\ttest_mpa:{:.5f}\ntest_class_acc :{:}'.format(
                train_miou / (len(test_data)), train_acc / (len(test_data)),
                train_mpa / (len(test_data)), train_class_acc / (len(test_data))))

            acc_list.append(train_acc / (len(test_data)))
            miou_list.append(train_miou / (len(test_data)))
            mpa_list.append(train_mpa / (len(test_data)))
            class_acc_list.append(train_class_acc / (len(test_data)))
            dice_list.append(dic[ii]+'.pth')

            time2 = time.time()
            v = time2 - time1

            print('The test {}'.format(dic[ii]))
            print(epoch_str)
            print("max_miou", max(miou_list))

            print("The test of {} test_elapsed_time:{}s".format(dic[ii], round(v, 1)))
            tt += v
            print("Already elapsed_time:{} min or {} h\n".format(round(tt / 60, 2), round(tt / 3600, 3)))

    name_ = ["-", "test_miou_list", "test_acc_list", "test_mpa_list", "test_class_acc_list"]
    list_ = [dice_list, miou_list, acc_list, mpa_list, class_acc_list]
    loss__ = pd.DataFrame(index=name_,data=list_)

    loss__.to_csv(os.path.join(base_dir, "test_epoch.csv"))             # 2021年6月26日09:23:05


