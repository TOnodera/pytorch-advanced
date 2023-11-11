from utils import ImageTransform, make_datapath_list, HymenopteraDataset
from torch.utils import data
from torchvision import models
from torch import nn, optim
from finetuning_train_model import train_model
import torch

# データ生成部
train_list = make_datapath_list(phase='train')
val_list = make_datapath_list(phase='val')

size = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
train_dataset = HymenopteraDataset(file_list=train_list, transform=ImageTransform(size, mean, std), phase='train')
val_dataset = HymenopteraDataset(file_list=val_list, transform=ImageTransform(size, mean, std), phase='val')

batch_size = 32

train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True) 

dataloaders_dict = {'train': train_dataloader, 'val': val_dataloader}


# ネットワーク設定部
use_pretrained = True
net = models.vgg16(pretrained=use_pretrained)
net.classifier[6] = nn.Linear(in_features=4096, out_features=2)
net.train()
print('ネットワーク設定完了：学習済みの重みデータをロードし、訓練モードに設定しました。')

# 損失関数設定部
criterion = nn.CrossEntropyLoss()

# ファインチューニングで学習させるパラメータを格納する
params_to_update_1 = [] # featuers層のパラメータ
params_to_update_2 = [] # classier層のパラメータ
params_to_update_3 = [] # 最後の全結合層（つけかえたやつ）のパラメータ

# 学習させる層のパラメータ名を指定
update_params_name_1 = ["features"]
update_params_name_2 = ["classifier.0.weight", "classifier.0.bias", "classifier.3.weight", "classifier.3.bias"]
update_params_name_3 = ["classifier.6.weight", "classifier.6.bias"]

# パラメータごとに各リストに格納する
for name, param in net.named_parameters():
    if update_params_name_1[0] in name:
        param.requires_grad = True
        params_to_update_1.append(param)
        print(f"param_to_update_1に格納: {name}")
        
    elif name in update_params_name_2:
        param.requires_grad = True
        params_to_update_2.append(param)
        print(f"param_to_update_2に格納: {name}")
        
    elif name in update_params_name_3:
        param.requires_grad = True
        params_to_update_3.append(param)
        print(f"param_to_update_3に格納: {name}")
        
    else:
        param.requires_grad = False
        

# 最適化手法の設定    
optimizer = optim.SGD([
    {'params': params_to_update_1, 'lr': 1e-4},
    {'params': params_to_update_2, 'lr': 5e-4},
    {'params': params_to_update_3, 'lr': 1e-3},
], momentum=0.9) 

num_epochs = 2
train_model(net, dataloaders_dict, criterion, optimizer, num_epochs)

# 学習した重みのセーブ
save_path = "./weight_fine_tuning.pth"
torch.save(net.state_dict(), save_path)
        

