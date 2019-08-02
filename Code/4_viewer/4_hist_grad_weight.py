# coding: utf-8

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import os
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import sys
import os
sys.path.append("..")
from utils.utils import MyDataset, validate, show_confMat, Net
from tensorboardX import SummaryWriter
from datetime import datetime

train_txt_path = os.path.join("..", "..", "Data", "train.txt")
valid_txt_path = os.path.join("..", "..", "Data", "valid.txt")

classes_name = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

train_bs = 16
valid_bs = 16
lr_init = 0.001
max_epoch = 1

# log
log_dir = os.path.join("..", "..", "Result", "hist_grad_weight")

writer = SummaryWriter(log_dir=log_dir)

# ------------------------------------ step 1/4 : 加载数据-------------------------------------------------

# 数据预处理设置
normMean = [0.4948052, 0.48568845, 0.44682974]
normStd = [0.24580306, 0.24236229, 0.2603115]
normTransform = transforms.Normalize(normMean, normStd)
trainTransform = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    normTransform
])

validTransform = transforms.Compose([
    transforms.ToTensor(),
    normTransform
])

# 构建MyDataset实例
train_data = MyDataset(txt_path=train_txt_path, transform=trainTransform)
valid_data = MyDataset(txt_path=valid_txt_path, transform=validTransform)

# 构建DataLoder
train_loader = DataLoader(dataset=train_data, batch_size=train_bs, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=valid_bs)

# ------------------------------------ step 2/4 : 网络初始化----------------------------------------------

net = Net()     # 创建一个网络
net.initialize_weights()    # 初始化权值

# ------------------------------------ step 3/4 : 定义损失函数和优化器 ------------------------------------

criterion = nn.CrossEntropyLoss()                                                   # 选择损失函数
optimizer = optim.SGD(net.parameters(), lr=lr_init, momentum=0.9, dampening=0.1)    # 选择优化器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)     # 设置学习率下降策略

# ------------------------------------ step 4/4 : 训练 --------------------------------------------------

for epoch in range(max_epoch):

    loss_sigma = 0.0    # 记录一个epoch的loss之和
    correct = 0.0
    total = 0.0
    scheduler.step()  # 更新学习率

    for i, data in enumerate(train_loader):
        # 获取图片和标签
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)

        # forward, backward, update weights
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 统计预测信息
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).squeeze().sum().numpy()
        loss_sigma += loss.item()

        # 每10个iteration 打印一次训练信息，loss为10个iteration的平均
        if i % 10 == 9:
            loss_avg = loss_sigma / 10
            loss_sigma = 0.0
            print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch + 1, max_epoch, i + 1, len(train_loader), loss_avg, correct / total))

    # 每个epoch，记录梯度，权值
    for name, layer in net.named_parameters():
        writer.add_histogram(name + '_grad', layer.grad.cpu().data.numpy(), epoch)
        writer.add_histogram(name + '_data', layer.cpu().data.numpy(), epoch)

print('Finished Training')
