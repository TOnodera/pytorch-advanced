from torchvision import transforms
import torch.utils.data as data
from PIL import Image
import torch.nn as nn
import torch
from torch import optim
import time

def make_datapath_list():

    train_img_list = list()

    for img_idx in range(200):
        img_path = "./data/img_78/img_7_" + str(img_idx) + ".jpg"
        train_img_list.append(img_path)

        img_path = "./data/img_78/img_8_" + str(img_idx) + ".jpg"
        train_img_list.append(img_path)

    return train_img_list

class ImageTransform:
    def __init__(self, mean, std) -> None:
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, img):
        return self.data_transform(img)

class GAN_Img_Dataset(data.Dataset):
    def __init__(self, filelist, transform) -> None:
        self.filelist = filelist
        self.transform = transform

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        img_path = self.filelist[index]
        img = Image.open(img_path) 

        img_transformed = self.transform(img)
        return img_transformed


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.01)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train_model(G, D, dataloader, num_epochs):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'使用デバイス: {device}')

    g_lr, d_lr  = 0.0001, 0.0004
    beta1, beta2 = 0.0, 0.9
    g_optimizer = optim.Adam(G.parameters(), g_lr,[beta1, beta2])
    d_optimizer = optim.Adam(D.parameters(), d_lr,[beta1, beta2])

    criterion = nn.BCEWithLogitsLoss(reduce='mean')

    z_dim = 20
    mini_batch_size = 64

    G.to(device)
    D.to(device)

    torch.backends.cudnn.benchmark = True

    num_train_imgs = len(dataloader.dataset)
    batch_size = dataloader.batch_size

    iteration = 1
    logs = []
    
    for epoch in range(num_epochs):
        t_epoch_start = time.time()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        print('-' * 10)
        print(f'Epoch {epoch}/{num_epochs}')
        print('-' * 10)
        print(' (train) ')


        for image in dataloader:
            if image.size()[0] == 1:
                continue

            ### Discriminatorの学習 ###
            image = image.to(device)

            # 正解ラベルと偽ラベルを作成
            mini_batch_size = image.size()[0]
            label_real = torch.full((mini_batch_size,), 1).to(device)
            label_fake = torch.full((mini_batch_size,), 0).to(device)

            # 真の画像を判定
            d_out_real = D(image)

            # 偽画像を生成して判定
            input_z = torch.randn(mini_batch_size, z_dim).to(device)
            input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
            fake_image = G(input_z)
            d_out_fake = D(fake_image)

            # 誤差を産出
            d_loss_real = criterion(d_out_real.view(-1), label_real.float())
            d_loss_fake = criterion(d_out_fake.view(-1), label_fake.float())
            d_loss = d_loss_real + d_loss_fake

            # バックプロパゲーション
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step() # Discriminatorについて学習を進行させている

            ### Generatorの学習 ###
            input_z = torch.randn(mini_batch_size, z_dim).to(device)
            input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
            fake_image = G(input_z)
            d_out_fake = D(fake_image)

            # 誤差を計算
            g_loss = criterion(d_out_fake.view(-1), label_real.float())

            # バックプロパゲーション
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step() # Generatorについて学習を進行させている

            # 記録
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            iteration += 1
        
        t_epoch_finish = time.time()
        print('-' * 10)
        print(f'epoch {epoch} || Epoch_D_Loss: {epoch_d_loss/batch_size:.4} || Epoch_G_Loss: {epoch_g_loss/batch_size:.4f}')
        print(f'timer: {(t_epoch_finish-t_epoch_start):.4f} sec')

    return G, D


