# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------- load_state_dict

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

    def zero_param(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.constant_(m.weight.data, 0)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.constant_(m.weight.data, 0)
                m.bias.data.zero_()
net = Net()

# 保存，并加载模型参数(仅保存模型参数)
torch.save(net.state_dict(), 'net_params.pkl')   # 假设训练好了一个模型net
pretrained_dict = torch.load('net_params.pkl')

# 将net的参数全部置0，方便对比
net.zero_param()
net_state_dict = net.state_dict()
print('conv1层的权值为:\n', net_state_dict['conv1.weight'], '\n')

# 通过load_state_dict 加载参数
net.load_state_dict(pretrained_dict)
print('加载之后，conv1层的权值变为:\n', net_state_dict['conv1.weight'])
