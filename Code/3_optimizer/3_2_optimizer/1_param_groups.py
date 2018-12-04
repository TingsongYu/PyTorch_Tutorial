# coding: utf-8

import torch
import torch.optim as optim


w1 = torch.randn(2, 2)
w1.requires_grad = True

w2 = torch.randn(2, 2)
w2.requires_grad = True

w3 = torch.randn(2, 2)
w3.requires_grad = True

# 一个参数组
optimizer_1 = optim.SGD([w1, w3], lr=0.1)
print('len(optimizer.param_groups): ', len(optimizer_1.param_groups))
print(optimizer_1.param_groups, '\n')

# 两个参数组
optimizer_2 = optim.SGD([{'params': w1, 'lr': 0.1},
                         {'params': w2, 'lr': 0.001}])
print('len(optimizer.param_groups): ', len(optimizer_2.param_groups))
print(optimizer_2.param_groups)
