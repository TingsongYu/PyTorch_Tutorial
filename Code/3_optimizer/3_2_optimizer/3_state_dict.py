# coding: utf-8

import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------- state_dict
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 1, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1 * 3 * 3, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 1 * 3 * 3)
        x = F.relu(self.fc1(x))
        return x


net = Net()

# 获取网络当前参数
net_state_dict = net.state_dict()

print('net_state_dict类型：', type(net_state_dict))
print('net_state_dict管理的参数: ', net_state_dict.keys())
for key, value in net_state_dict.items():
    print('参数名: ', key, '\t大小: ',  value.shape)
