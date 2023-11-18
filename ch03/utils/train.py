import torch
from .pspnet import *
import time
import pandas as pd

def train_model(net, dataloaders_dict, criterion, scheduler, optimizer, num_epochs):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")
    
    net.to(device)
    torch.backends.cudnn.benchmark=True
    
    # 画像の枚数
    num_train_imgs = len(dataloaders_dict["train"].dataset)
    num_val_imgs = len(dataloaders_dict["val"].dataset)
    batch_size = dataloaders_dict["train"].batch_size
    
    iteration = 1
    logs = []
    batch_multiplier = 3

    for epoch in range(num_epochs):

        t_epoch_start = time.time()
        t_iter_start = time.time()
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0
        
        print('----------')
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('----------')
        
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
                if count == 0:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                print(' (train) ')
            else:
                if (epoch + 1) % 5 == 0:
                    net.val()
                    print('----------')
                    print(' (train) ')
                else:
                    continue
            
            count = 0
            for images, anno_class_images in dataloaders_dict[phase]:
                if images.size() == 1:
                    continue
                
                images = images.to(device)
                anno_class_images = anno_class_images.to(device)
                
                if phase == 'train' and count == 0:
                    optimizer.zero_grad()
                    count = batch_multiplier

                # 順伝搬の計算
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(images)
                    loss = criterion(
                        outputs, anno_class_images.long()
                    ) / batch_multiplier

                    # 訓練時はバックプロぱげーしょん
                    if phase == 'train':
                        loss.backward()
                        count -= 1

                        if iteration % 10 == 0:
                            t_iter_finish = time.time()
                            duration = t_iter_finish - t_iter_finish
                            print(f'イテレーション {iteration} || Loss: {loss.item()/batch_size*batch_multiplier:.4f} || 10iter {duration}')
                            t_iter_start = time.time()
                        
                        epoch_train_loss += loss.item() * batch_multiplier
                        iteration += 1
                    
                    else:
                        epoch_val_loss += loss.item() * batch_multiplier

        t_epoch_finish = time.time()
        print('----------')
        print(f'epoch {epoch+1} || Epoch_Train_Loss: {epoch_train_loss/num_train_imgs:.4f} || Epoch_Val_Loss: {epoch_val_loss/num_val_imgs}')
        print(f'timer: {t_epoch_finish-t_epoch_start:.4f} sec')
        print('----------') 
        
        log_epoch = {'epoch': epoch+1, 'train_loss': epoch_train_loss/num_train_imgs, 'val_loss': epoch_val_loss/num_val_imgs}
        logs.append(log_epoch)
        
        df = pd.DataFrame(logs)
        df.to_csv('log_output.csv')
    
    torch.save(net.state_dict(), 'weights/pspnet50_' + str(epoch+1) + '.pth')
                
