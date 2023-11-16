from utils.dataloader import make_datapath_list, VOCDataset, DataTransform
from torch.utils import data

rootpath = "./data/VOCdevkit/VOC2012/"
train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(rootpath)

color_mean = (0.485, 0.456, 0.406)
color_std = (0.229, 0.224, 0.225)

train_dataset = VOCDataset(train_img_list, train_anno_list, phase='train',
                        transform=DataTransform(input_size=475, color_mean=color_mean, color_std=color_std))

val_dataset = VOCDataset(val_img_list, val_anno_list, phase='val',
                        transform=DataTransform(input_size=475, color_mean=color_mean, color_std=color_std))

batch_size = 8

train_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

dataloader_dict = {'train': train_dataloader, 'val': val_dataloader}

batch_iterator = iter(dataloader_dict['val'])
images, anno_class_images = next(batch_iterator)
print(images.size())
print(anno_class_images.size())