import torch.utils.data.dataset as data
from typing import Any
import cv2
import torch
import numpy as np

class VOCDataset(data.Dataset):
    """
    VOC2012のDatasetを格納する
    
    Attributes
    ----------

    img_list: List
       画像のパスを格納したリスト
    anno_list: List
        アノテーションのパスを格納したリスト
    phase: 'train' or 'test'
        学習か訓練かを設定する
    transform: object
        前処理クラスのインスタンス
    transform_anno:
        xmlのアノテーションをリストに変換するインスタンス
    """
    def __init__(self, img_list, anno_list, phase, transform, transform_anno) -> None:
        self.img_list = img_list
        self.anno_list = anno_list
        self.pahse = phase
        self.transform = transform
        self.transform_anno = transform_anno
        
    def __len__(self):
        """
        画像の枚数を返す
        """
        return len(self.img_list)
    
    def __getitem__(self, index) -> Any:
        im, gt, _, _ = self.pull_item(index)
        return im, gt

    def pull_item(self, index):
        """
        前処理をした画像のテンソル形式のデータ、アノテーション、画像の高さ、幅を取得する
        """
        
        # 画像の読み込み
        image_file_path = self.img_list[index]
        img = cv2.imread(image_file_path)
        height, width, channels = img.shape
        
        # xml形式のアノテーション情報をリストに
        anno_file_path = self.anno_list[index]
        anno_list = self.transform_anno(anno_file_path, width, height)

        # 前処理の実施
        img, boxes, labels = self.transform(img, self.pahse, anno_list[:, :4], anno_list[:, 4])
        
        # 色チャネルの順番がBGRになっているので、RGBに順番を変更
        # さらに（高さ、幅、色チャネル）の順を（色チャネル、高さ、幅）に変更
        img = torch.from_numpy(img[:, :, (2,1,0)]).permute(2,0,1)

        # BBoxとラベルをセットにしたnp.arrayを作成
        gt = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        
        return img, gt, height, width
    
    
    