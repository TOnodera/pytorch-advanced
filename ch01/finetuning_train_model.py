from tqdm import tqdm
import torch

def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):

    # デバイス設定(GPUが使える場合はGPUで実行する)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    print(f'使用デバイス: {device}')
    
    # ネットワークがある程度固定であれば高速化させる
    torch.backends.cudnn.benchmark = True
    

    # epochループ
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-----------------------------')
        
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()
            
            epoch_loss = 0.0
            epoch_corrects = 0

            if epoch == 0 and phase == 'train':
                continue
            
            for inputs, labels in tqdm(dataloaders_dict[phase]):
                
                # GPUが使える場合はGPUで実行する
                inputs = inputs.to(device)
                labels = labels.to(device)

                # optimizerの初期化
                optimizer.zero_grad()

                # forward計算
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    
                    # 訓練時は誤差逆伝搬
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # イテレーション結果の計算
                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)
                
            # epochごとのlossと正解率を表示
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
