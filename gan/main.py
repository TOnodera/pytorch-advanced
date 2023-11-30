import matplotlib.pyplot as plt
from generator import Generator
import torch
from discriminator import Discriminator
import torch.nn as nn
import utils
import torch.utils.data as data

G = Generator()
input_z = torch.randn(1, 20)
input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)

fake_image = G(input_z)
D = Discriminator()

G.apply(utils.weights_init)
D.apply(utils.weights_init)

print('ネットワーク初期化完了')

train_img_list = utils.make_datapath_list()
mean = (0.5,)
std = (0.5,)
train_dataset = utils.GAN_Img_Dataset(filelist=train_img_list, transform=utils.ImageTransform(mean, std))
batch_size = 64
train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
num_epochs = 200

G_updated, D_updated = utils.train_model(G, D, dataloader=train_dataloader, num_epochs=num_epochs)

device = torch.device('cuda:0')

batch_size = 8
z_dim = 20
fixed_z = torch.randn(batch_size, z_dim)
fixed_z = fixed_z.view(fixed_z.size(0), fixed_z.size(1), 1, 1)

# 訓練したジェネレータで画像を生成
fake_image = G_updated(fixed_z.to(device))

# 訓練データを取得
batch_iterator = iter(train_dataloader)
imgs = next(batch_iterator)

fig = plt.figure(figsize=(15, 6))
for i in range(0, 5):
    plt.subplot(2, 5, i+1)
    plt.imshow(imgs[i][0].cpu().detach().numpy(), 'gray')

    plt.subplot(2, 5, 5+i+1)
    plt.imshow(fake_image[i][0].cpu().detach().numpy(), 'gray')

plt.show()


