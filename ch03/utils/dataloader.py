from utils.data_augumentation import Compose, Scale, RandomRotation, RandomMirror, Resize, Normalize_Tensor
import os.path as osp
from torch.utils import data
from PIL import Image
from typing import Any


def make_datapath_list(rootpath):
    """
    学習データ、アノテーションデータ、検証データへのパスリストを作成
    """
    imgpath_template = osp.join(rootpath, "JPEGImages", "%s.jpg")
    annopath_template = osp.join(rootpath, "SegmentationClass", "%s.png")
    
    train_id_names = osp.join(rootpath + "ImageSets/Segmentation/train.txt")
    val_id_names = osp.join(rootpath + "ImageSets/Segmentation/val.txt")
    
    train_img_list = list()
    train_anno_list = list()
    
    for line in open(train_id_names):
        file_id = line.strip()
        img_path = (imgpath_template % file_id)
        anno_path = (annopath_template % file_id)
        train_img_list.append(img_path)
        train_anno_list.append(anno_path)
        

    val_img_list = list()
    val_anno_list = list()
    
    for line in open(val_id_names):
        file_id = line.strip()
        img_path = (imgpath_template % file_id)
        anno_path = (annopath_template % file_id)
        val_img_list.append(img_path)
        val_anno_list.append(anno_path)
        
    return train_img_list, train_anno_list, val_img_list, val_anno_list

class DataTransform:
    def __init__(self, input_size, color_mean, color_std) -> None:
        self.data_transform = {
            'train': Compose([
                Scale(scale=[0.5, 1.5]),
                RandomRotation(angle=[0.5, 1.5]),
                RandomMirror(),
                Resize(input_size),
                Normalize_Tensor(color_mean, color_std)
            ]),
            'val': Compose([
                Resize(input_size),
                Normalize_Tensor(color_mean, color_std)
            ])
        }
        
    def __call__(self, phase, img, anno_class_img):
        return self.data_transform[phase](img, anno_class_img)
    
class VOCDataset(data.Dataset):
    def __init__(self, img_list, anno_list, phase, transform) -> None:
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase 
        self.transform = transform
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index) -> Any:
        return self.pull_item(index)

    def pull_item(self, index):
        image_file_path = self.img_list[index]
        img = Image.open(image_file_path) # [高さ, 幅, 色]
        
        anno_file_path = self.anno_list[index]
        anno_class_img = Image.open(anno_file_path) # [高さ, 幅]
        
        # 前処理を実施
        img, anno_class_img = self.transform(self.phase, img, anno_class_img)
        
        return img, anno_class_img







