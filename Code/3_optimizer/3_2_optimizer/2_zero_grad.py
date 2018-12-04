# coding: utf-8

import torch
import torch.optim as optim

# ----------------------------------- zero_grad

w1 = torch.randn(2, 2)
w1.requires_grad = True

w2 = torch.randn(2, 2)
w2.requires_grad = True

optimizer = optim.SGD([w1, w2], lr=0.001, momentum=0.9)

optimizer.param_groups[0]['params'][0].grad = torch.randn(2, 2)

print('参数w1的梯度：')
print(optimizer.param_groups[0]['params'][0].grad, '\n')  # 参数组，第一个参数(w1)的梯度

optimizer.zero_grad()
print('执行zero_grad()之后，参数w1的梯度：')
print(optimizer.param_groups[0]['params'][0].grad)  # 参数组，第一个参数(w1)的梯度
