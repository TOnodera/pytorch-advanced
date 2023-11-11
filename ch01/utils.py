import glob
import os.path as osp
from typing import Any
from torchvision import transforms
import torch.utils.data as data
from PIL import Image

def make_datapath_list(phase="train"):
    root_path = "./data/hymenoptera_data/"
    target_path = osp.join(root_path+phase+'/**/*.jpg')
    
    path_list = []
    
    for path in glob.glob(target_path):
        path_list.append(path)

    return path_list

class ImageTransform:
    """
    画像の前処理クラス。訓練時、検証時で異なる動作をする。
    画像のサイズをリサイズし、色を標準化する。
    訓練時はRandomResizedCropとRandomHorizontalFlipでデータオーギュメンテーションする。

    Attribute:
    ----------
    resize: int
        リサイズ先の画像の大きさ
    mean: (R, G, B)
        各色チャネルの平均値
    std: (R, G, B)
        各色チャネルの標準偏差
    """
    def __init__(self, resize, mean, std) -> None:
        self.data_transform = {
            "train": 
                transforms.Compose([
                    # scaleで指定した範囲のなかで画像を拡大・縮小し、アスペクト比を3/4-4/3で出力する
                    transforms.RandomResizedCrop(resize, scale=(0.5,1.0)),
                    # 50%の確率で画像を左右反転させる
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
            ]),
            "val":  
                transforms.Compose([
                    transforms.Resize(resize),
                    transforms.CenterCrop(resize),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
            ])

        }
        
    def __call__(self, img, phase='train') -> Any:
        """
        
        Parameters
        ----------
        phase: 'train' or 'val'
        """
        return self.data_transform[phase](img)

class HymenopteraDataset(data.Dataset):
    """
    アリとハチの画像のDatasetクラス。PytorchのDatasetクラスを継承。
    
    Attributes
    ----------
    file_list: List
        画像のパスを格納したリスト。
    transform: object
        前処理クラスのインスタンス
    phase: 'train' or 'test'
        学習か訓練かを設定する。
    """
    def __init__(self, file_list, transform, phase='train') -> None:
        self.file_list = file_list
        self.transform = transform
        self.phase = phase
        
    def __len__(self):
        """
        画像の枚数を返す
        """
        return len(self.file_list)
    
    def __getitem__(self, index):
        """
        前処理をした画像のTensor形式のデータとラベルを取得
        """
        # index番目の画像をロード
        img_path = self.file_list[index]
        img = Image.open(img_path)
        
        # 画像の前処理を実施
        img_transformed = self.transform(img, self.phase)

        # 画像のラベルをファイル名から抜き出す
        if self.phase == 'train':
            label = img_path[30:34]
        elif self.phase == 'val':
            label = img_path[28:32]

        if label == "ants":
            label = 0
        elif label == "bees":
            label = 1
        
        return img_transformed, label