# coding: utf-8
import os
import torch
import torchvision.utils as vutils
import numpy as np
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import torchvision.transforms as transforms
import sys
sys.path.append("..")
from utils.utils import MyDataset, Net, normalize_invert
from torch.utils.data import DataLoader


vis_layer = 'conv1'
log_dir = os.path.join("..", "..", "Result", "visual_featuremaps")
txt_path = os.path.join("..", "..", "Data", "visual.txt")
pretrained_path = os.path.join("..", "..", "Data", "net_params_72p.pkl")

net = Net()
pretrained_dict = torch.load(pretrained_path)
net.load_state_dict(pretrained_dict)

# 数据预处理
normMean = [0.49139968, 0.48215827, 0.44653124]
normStd = [0.24703233, 0.24348505, 0.26158768]
normTransform = transforms.Normalize(normMean, normStd)
testTransform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    normTransform
])
# 载入数据
test_data = MyDataset(txt_path=txt_path, transform=testTransform)
test_loader = DataLoader(dataset=test_data, batch_size=1)
img, label = iter(test_loader).next()

x = img
writer = SummaryWriter(log_dir=log_dir)
for name, layer in net._modules.items():

    # 为fc层预处理x
    x = x.view(x.size(0), -1) if "fc" in name else x

    # 对x执行单层运算
    x = layer(x)
    print(x.size())

    # 由于__init__()相较于forward()缺少relu操作，需要手动增加
    x = F.relu(x) if 'conv' in name else x

    # 依据选择的层，进行记录feature maps
    if name == vis_layer:
        # 绘制feature maps
        x1 = x.transpose(0, 1)  # C，B, H, W  ---> B，C, H, W
        img_grid = vutils.make_grid(x1, normalize=True, scale_each=True, nrow=2)  # B，C, H, W
        writer.add_image(vis_layer + '_feature_maps', img_grid, global_step=666)

        # 绘制原始图像
        img_raw = normalize_invert(img, normMean, normStd)  # 图像去标准化
        img_raw = np.array(img_raw * 255).clip(0, 255).squeeze().astype('uint8')
        writer.add_image('raw img', img_raw, global_step=666)  # j 表示feature map数
writer.close()