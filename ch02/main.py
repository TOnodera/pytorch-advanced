import utils
from anno_xml2list import Anno_xml2list
import cv2
import matplotlib.pyplot as plt
from data_transform import DataTransform
import numpy as np
from voc_dataset import VOCDataset
from torch.utils.data.dataloader import DataLoader



rootpath = "./data/VOCdevkit/VOC2012/"
train_img_list, train_anno_list, val_img_list, val_anno_list = utils.make_datapath_list(rootpath)

# 動作確認　
voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

transform_anno = Anno_xml2list(voc_classes)

# 画像の読み込み OpenCVを使用
image_file_path = train_img_list[0]
img = cv2.imread(image_file_path)  # [高さ][幅][色BGR]
height, width, channels = img.shape  # 画像のサイズを取得

# アノテーションをリストで表示
anno_list = transform_anno(train_anno_list[0], width, height)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.show()

color_mean = (104, 117, 123)
input_size = 300
transform = DataTransform(input_size, color_mean)
phase = 'train'
img_transformed, boxes, labels = transform(img, phase, anno_list[:, :4], anno_list[:, 4])
plt.imshow(cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB))
# plt.show()

train_dataset = VOCDataset(train_img_list, train_anno_list, phase='train', transform=DataTransform(input_size, color_mean), transform_anno=Anno_xml2list(voc_classes))
val_dataset = VOCDataset(val_img_list, val_anno_list, phase='val', transform=DataTransform(input_size, color_mean), transform_anno=Anno_xml2list(voc_classes))

batch_size = 4
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=utils.od_collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=utils.od_collate_fn)
dataloaders_dict = {
    "train": train_dataloader,
    "val": val_dataloader
}

batch_iterator = iter(dataloaders_dict["val"])
images, targets = next(batch_iterator)

print(train_dataset.__len__())
print(val_dataset.__len__())

