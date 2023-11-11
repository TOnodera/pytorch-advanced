import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import models
from utils import make_datapath_list
from hymenoptera_dataset import HymenopteraDataset
import utils
from image_transform import ImageTransform
from train_model import train_model

# シード設定
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

train_list = utils.make_datapath_list('train')
val_list = utils.make_datapath_list('val')

size = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

train_dataset = HymenopteraDataset(
    file_list=train_list, 
    transform=ImageTransform(size, mean, std),
    phase='train'
)
val_dataset = HymenopteraDataset(
    file_list=val_list, 
    transform=ImageTransform(size, mean, std),
    phase='val'
)

# バッチサイズ
batch_size = 32

# DataLoader生成
train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 辞書型変数にまとめる
dataloaders_dict = {
    'train': train_dataloader,
    'val': val_dataloader
}

# 学習済みのVGG-16モデルのインスタンスを生成
use_pretrained = True
net = models.vgg16(pretrained=use_pretrained)

# VGG16の最後の出力層を2個にする
net.classifier[6] = nn.Linear(in_features=4096, out_features=2)

# 訓練モードに設定
net.train()
print('ネットワーク設定完了: 学習の重みをロードし、訓練モードに設定しました。')

criterion = nn.CrossEntropyLoss()
# 最適化手法を設定
params_to_update = []
# 学習パラメータ名
update_param_names = ["classifier.6.weight", "classifier.6.bias"]

# 学習させるパラメータ以外は勾配計算をなくし、変化しないように設定
for name, param in net.named_parameters():
    if name in update_param_names:
        param.requires_grad = True
        params_to_update.append(param)
    else:
        param.requires_grad = False
        
optimizer = optim.SGD(params=params_to_update, lr=0.001, momentum=0.9)

num_epochs = 2
train_model(net, dataloaders_dict, criterion, optimizer, num_epochs)

