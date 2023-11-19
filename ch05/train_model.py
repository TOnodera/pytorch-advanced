import torch
from torch import nn, optim
import time

def train_model(G, D, dataloader, num_epochs):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'使用デバイス: {device}')
    
    # 最適化手法の設定
    g_lr, d_lr = 0.001, 0.004
    beta1, beta2 = 0.0, 0.9
    d_optimizer = optim.Adam(G.parameters(), g_lr, [beta1, beta2])
    g_optimizer = torch.optim.Adam(D.parameters(), d_lr, [beta1, beta2])
    
    # 誤差関数を定義
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    
    # パラメータをハードコーディング
    z_dim = 20
    mini_batch_size = 64

    G.to(device)
    D.to(device)
    
    G.train()
    D.train()
    
    torch.backends.cudnn.benchmark = True
    
    num_train_imgs = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    
    iteration = 1
    logs = []

    for epoch in range(num_epochs):
        t_epoch_start = time.time()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        
        print('----------')
        print(f'Epoch {epoch}/{num_epochs}')
        print('----------')
        print(' (train) ')
        
        for images in dataloader:
            
            ### Discriminatorの学習
            if images.size()[0] == 1:
                continue
                
            images = images.to(device)
            
            mini_batch_size = images.size()[0]
            label_real = torch.full((mini_batch_size,), 1).to(device)
            label_fake = torch.full((mini_batch_size,), 0).to(device)
            
            d_out_real = D(images)
            
            # 偽画像を生成して判定
            input_z = torch.randn(mini_batch_size, z_dim).to(device)
            input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
            fake_images = G(input_z)
            d_out_fake = D(fake_images)
            
            # 誤差を計算
            d_loss_real = criterion(d_out_real.view(-1), label_real.float())
            d_loss_fake = criterion(d_out_fake.view(-1), label_fake.float())
            d_loss = d_loss_real + d_loss_fake
            
            # バックプロパゲーション
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()
            
            d_loss.backward()
            d_optimizer.step()
            
            ### Generatorの学習
            input_z = torch.randn(mini_batch_size, z_dim).to(device)
            input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
            fake_images = G(input_z)
            d_out_fake = D(fake_images)
            
            g_loss = criterion(d_out_fake.view(-1), label_real.float())
            
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()

            g_loss.backward()
            g_optimizer.step()
            
            # 記録
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            iteration += 1

        t_epoch_finish = time.time()
        print('----------')
        print(f'Epoch {epoch} || Epoch_D_Loss: {epoch_d_loss/batch_size:.4f} || Epoch_G_Loss: {epoch_g_loss/batch_size:.4f}')
        print(f'timer: {t_epoch_finish-t_epoch_start:.4f} sec')
        print('----------')
        t_epoch_start = time.time()
        
    return G, D

