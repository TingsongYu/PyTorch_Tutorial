# coding: utf-8

import torch
import torch.nn as nn
import numpy as np

# ----------------------------------- log likelihood loss

# 各类别权重
weight = torch.from_numpy(np.array([0.6, 0.2, 0.2])).float()

# 生成网络输出 以及 目标输出
output = torch.from_numpy(np.array([[0.7, 0.2, 0.1], [0.4, 1.2, 0.4]])).float()  
output.requires_grad = True
target = torch.from_numpy(np.array([0, 0])).type(torch.LongTensor)


loss_f = nn.NLLLoss(weight=weight, size_average=True, reduce=False)
loss = loss_f(output, target)

print('\nloss: \n', loss)
print('\n第一个样本是0类，loss = -(0.6*0.7)={}'.format(loss[0]))
print('\n第二个样本是0类，loss = -(0.6*0.4)={}'.format(loss[1]))