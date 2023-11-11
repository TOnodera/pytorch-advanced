from typing import Any
from torchvision import transforms
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