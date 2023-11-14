import torch
import time
from torch import nn
import pandas as pd

def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")
    
    net.to(device)
    
    # イテレーションカウンタセット
    iteration = 1
    epoch_train_loss = 0.0
    epoch_val_loss = 0.0
    logs = []

    # epochループ
    for epoch in range(num_epochs + 1):

        # 開始時刻を保存
        t_epoch_start = time.time()
        t_iter_start = time.time()
        
        print('----------')
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('----------')
        
        # epochごとの訓練と検証のループ 
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
                print(' (train) ')
            else:
                if (epoch + 1) % 10 == 0:
                    net.eval()
                    print('----------')
                    print(' (val) ')
                else:
                    continue
            
            for images, targets in dataloaders_dict[phase]:
                images = images.to(device)
                targets = [ann.to(device) for ann in targets]
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    outputs = net(images)

                    loss_l, loss_c =  criterion(outputs, targets)
                    loss = loss_l + loss_c

                    # 訓練時はバックプロパゲーション
                    if phase == 'train':
                        loss.backward()
                        # 勾配爆発がおきないようにクリッピングする
                        nn.utils.clip_grad_value_(net.parameters(), clip_value=2.0)
                        # パラメータ更新
                        optimizer.step()

                        if iteration % 10 == 0:
                            t_iter_finish = time.time()
                            duration = t_iter_finish - t_iter_start
                            print(f'イテレーション {iteration} || Loss: {loss.item():.4f} || 10iter: {duration:.4f}')
                        
                        epoch_train_loss += loss.item()
                        iteration += 1
                    
                    else:
                        # 検証時
                        epoch_val_loss += loss.item()

        t_epoch_finish = time.time()
        print('----------')        
        print(f'epoch {epoch+1} || Epoch_Train_loss: {epoch_train_loss:.4f} || Epoch_Val_Loss: {epoch_val_loss:.4f}')
        print(f'timer: {t_epoch_finish - t_epoch_start:.4f}')
        t_epoch_start = time.time()
        
        # ログを保存
        log_epoch = {'epoch': epoch+1,
                     'train_loss': epoch_train_loss, 'val_loss': epoch_val_loss}
        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        df.to_csv("log_output.csv")

        epoch_train_loss = 0.0  # epochの損失和
        epoch_val_loss = 0.0  # epochの損失和

        # ネットワークを保存する
        if ((epoch+1) % 10 == 0):
            torch.save(net.state_dict(), 'weights/ssd300_' +
                       str(epoch+1) + '.pth')


