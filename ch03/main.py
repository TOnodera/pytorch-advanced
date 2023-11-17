from utils.dataloader import make_datapath_list, VOCDataset, DataTransform
from torch.utils import data
from utils import pspnet 

rootpath = "./data/VOCdevkit/VOC2012/"
train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(rootpath)

color_mean = (0.485, 0.456, 0.406)
color_std = (0.229, 0.224, 0.225)

train_dataset = VOCDataset(train_img_list, train_anno_list, phase='train',
                        transform=DataTransform(input_size=475, color_mean=color_mean, color_std=color_std))

val_dataset = VOCDataset(val_img_list, val_anno_list, phase='val',
                        transform=DataTransform(input_size=475, color_mean=color_mean, color_std=color_std))

batch_size = 8

train_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

dataloader_dict = {'train': train_dataloader, 'val': val_dataloader}

net = pspnet.PSPNet(n_classes=21)
batch_size = 2
import torch
dummy_img = torch.rand(batch_size, 3, 475, 475)
outputs = net(dummy_img)
print(outputs)