# coding: utf-8
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
import matplotlib.pyplot as plt

vis_layer = 'conv1'
log_dir = '../../Result/visual_featuremaps'
txt_path = '../../Data/visual.txt'
pretrained_path = '../../Data/net_params_72p.pkl'

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


# Visualize feature maps
features_dict = {}
def get_features(name):
    def hook(model, input, output):
        features_dict[name] = output.detach()
    return hook


net.conv1.register_forward_hook(get_features('ext_conv1'))
output = net(x)

features = features_dict['ext_conv1'].view(-1, 1, 28, 28)

img = vutils.make_grid(features, normalize=True, scale_each=True, nrow=3)
plt.imshow(img.permute(1, 2, 0))
plt.axis('off')
plt.show()
