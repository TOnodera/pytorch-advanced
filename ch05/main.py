import matplotlib.pyplot as plt
from generator import Generator
from discriminator import Discriminator
from torch.utils.data import DataLoader
from torch import nn
from utils import utils
from train_model import train_model
import torch


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        


G = Generator(z_dim=20, image_size=64) 
D = Discriminator(z_dim=20, image_size=64)
# 重みの初期化
G.apply(weight_init)
D.apply(weight_init)
train_img_list = utils.make_datapath_list()

mean = (0.5,)
std = (0.5,)

train_dataset = utils.GAN_Img_Dataset(file_list=train_img_list, transform=utils.ImageTransform(mean, std))

batch_size = 64

train_dataloader= DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

num_epoch = 500
G_update, D_update = train_model(G, D, dataloader=train_dataloader, num_epochs=num_epoch)

# 学習したパラメータを用いて画像を生成してみる
batch_size = 8
z_dim = 20
fixed_z = torch.randn(batch_size, z_dim)
fixed_z = fixed_z.view(fixed_z.size(0), fixed_z.size(1), 1, 1)
fake_images = G_update(fixed_z.to('cuda:0'))

batch_iterator = iter(train_dataloader)
images = next(batch_iterator)

fig = plt.figure(figsize=(15,6))
for i in range(0, 5):
    plt.subplot(2, 5, i+1)
    plt.imshow(images[i][0].cpu().detach().numpy(), 'gray')
    
    plt.subplot(2, 5, 5+i+1)
    plt.imshow(fake_images[i][0].cpu().detach().numpy(), 'gray')

plt.show()    