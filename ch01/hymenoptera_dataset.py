import torch.utils.data as data
from PIL import Image

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