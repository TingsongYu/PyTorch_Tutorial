# coding: utf-8
"""
    将原始数据集进行划分成训练集、验证集和测试集
"""

import os
import glob
import random
import shutil

dataset_dir = os.path.join("..", "..", "Data", "cifar-10-png", "raw_test")
train_dir = os.path.join("..", "..", "Data", "train")
valid_dir = os.path.join("..", "..", "Data", "valid")
test_dir = os.path.join("..", "..", "Data", "test")

train_per = 0.8
valid_per = 0.1
test_per = 0.1


def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


if __name__ == '__main__':

    for root, dirs, files in os.walk(dataset_dir):
        for sDir in dirs:
            imgs_list = glob.glob(os.path.join(root, sDir, '*.png'))
            random.seed(666)
            random.shuffle(imgs_list)
            imgs_num = len(imgs_list)

            train_point = int(imgs_num * train_per)
            valid_point = int(imgs_num * (train_per + valid_per))

            for i in range(imgs_num):
                if i < train_point:
                    out_dir = os.path.join(train_dir, sDir)
                elif i < valid_point:
                    out_dir = os.path.join(valid_dir, sDir)
                else:
                    out_dir = os.path.join(test_dir, sDir)

                makedir(out_dir)
                out_path = os.path.join(out_dir, os.path.split(imgs_list[i])[-1])
                shutil.copy(imgs_list[i], out_path)

            print('Class:{}, train:{}, valid:{}, test:{}'.format(sDir, train_point, valid_point-train_point, imgs_num-valid_point))
