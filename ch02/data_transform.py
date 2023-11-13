from typing import Any
from data_augumentation import Compose, ConvertFromInts, ToAbsoluteCoords, PhotometricDistort, Expand, RandomSampleCrop, RandomMirror, ToPercentCoords, Resize, SubtractMeans

class DataTransform:
    def __init__(self, input_size, color_mean) -> None:
        self.data_transform = {
                'train': Compose([
                    ConvertFromInts(), # intをfloat32に変換
                    ToAbsoluteCoords(), # アノテーションデータの規格化を戻す
                    PhotometricDistort(), # 画像の色調をランダムに変化
                    Expand(color_mean), # 画像のキャンバスを広げる
                    RandomSampleCrop(), # 画像内の部分をランダムに抜き出す
                    RandomMirror(), # 画像を反転させる
                    ToPercentCoords(), # アノテーションを0-1に規格化
                    Resize(input_size),
                    SubtractMeans(color_mean), # BGRの平均値を減算
                ]),
                'val': Compose([
                    ConvertFromInts(),
                    Resize(input_size),
                    SubtractMeans(color_mean)
                ])
        }
        
    def __call__(self, img, phase, boxes, labels) -> Any:
        return self.data_transform[phase](img, boxes, labels)