import os.path as osp
import torch
from torch import nn

def make_datapath_list(rootpath):
    """
    データのパスを格納したリストを返す

    Parameters:
    ----------
    rootpath: str
        データフォルダへのパス
    
    Returns
    ----------
    ret: train_img_list, train_anno_list, val_img_list, val_anno_list
        データへのパスを格納したリスト
    """
    
    imagepath_template = osp.join(rootpath, 'JPEGImages', '%s.jpg')
    annopath_template = osp.join(rootpath, 'Annotations', '%s.xml')
    
    train_id_names = osp.join(rootpath + 'ImageSets/Main/train.txt')
    val_id_names = osp.join(rootpath + 'ImageSets/Main/val.txt')
    
    train_img_list = list()
    train_anno_list = list()
   
    for line in open(train_id_names):
        file_id = line.strip()
        img_path = (imagepath_template % file_id)
        anno_path = (annopath_template % file_id)
        train_img_list.append(img_path)
        train_anno_list.append(anno_path)
        
    val_img_list = list()
    val_anno_list = list()

    for line in open(val_id_names):
        file_id = line.strip()
        img_path = (imagepath_template % file_id)
        anno_path = (annopath_template % file_id)
        val_img_list.append(img_path)
        val_anno_list.append(anno_path)

    return train_img_list, train_anno_list, val_img_list, val_anno_list

def od_collate_fn(batch):
    """
    Datasetから取り出すアノテーションデータのサイズが画像ごとに異なります。
    画像内の物体数が2個であれば（２，５）というサイズですが、3個であれば（３，５）などと変化します。
    この変化に対応したDataloaderを作成するためにcollate_fnは、PyTorchでリストからmini-batchを作成する関数です。
    ミニバッチ分の画像が並んでいるリスト変数batchに、ミニバッチの番号を指定する次元を先頭に一つ追加してリストの形を変形します。
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0]) # sample[0]はimg
        targets.append(torch.FloatTensor(sample[1])) # sample[1]はアノテーションデータgt

    
    # imgsはミニバッチサイズのリストになっています
    # リストの要素はtorch.Size([3, 300, 300])です。
    # このリストをtorch.Size([batch_num, 3, 300, 300])
    # のテンソルに変換します。
    imgs = torch.stack(imgs, dim=0)

def make_vgg():
    """
    34層vggモジュールを生成
    """
    layers = []
    in_channels = 3 # 色チャネル数

    cfg = [64, 64, 'M', 128 ,128, 'M', 256, 256, 256, 'MC', 512, 512, 512, 'M', 512, 512, 512]

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'MC':
            # ceil_modeでfloatに対して切り上げるようにTrueを設定、デフォルトは切り捨てる
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)] # inplaceはTrueにするとRelu関数の入力をメモリに保持しないのでリソース節約できる
            in_channels = v
        
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]

    return nn.ModuleList(layers)

def make_extras():
    """
    8層のextrasモジュールを生成
    """
    layers = []
    in_channels = 1024
    
    cfg = [256, 512, 128, 256, 128, 256, 128, 256]

    layers += [nn.Conv2d(in_channels, cfg[0], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[0], cfg[1], kernel_size=(3), stride=2, padding=1)]
    layers += [nn.Conv2d(cfg[1], cfg[2], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[2], cfg[3], kernel_size=(3), stride=2, padding=1)]
    layers += [nn.Conv2d(cfg[3], cfg[4], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[4], cfg[5], kernel_size=(3))]
    layers += [nn.Conv2d(cfg[5], cfg[6], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[6], cfg[7], kernel_size=(3))]
    
    return nn.ModuleList(layers)


def make_loc_conf(num_classes=21, bbox_aspect_num=[4,6,6,6,4,4]):
    # デフォルトボックスのオフセットを出力する
    loc_layers = []
    # デフォルトボックスに対する各クラスの信頼度confidenceを出力する
    conf_layers = []
    
    # VGGの22層目,conv4_3(source1)に対する畳み込み層
    loc_layers += [nn.Conv2d(512, bbox_aspect_num[0] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(512, bbox_aspect_num[0] * num_classes, kernel_size=3, padding=1)]
    
    # VGGの最終層(source2)に対する畳み込み層
    loc_layers += [nn.Conv2d(1024, bbox_aspect_num[1] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(1024, bbox_aspect_num[1] * num_classes, kernel_size=3, padding=1)]
    
    # extraの(source3)に対する畳み込み層
    loc_layers += [nn.Conv2d(512, bbox_aspect_num[2] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(512, bbox_aspect_num[2] * num_classes, kernel_size=3, padding=1)]

    # extraの(source4)に対する畳み込み層
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[3] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[3] * num_classes, kernel_size=3, padding=1)]
    
    # extraの(source5)に対する畳み込み層
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[4] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[4] * num_classes, kernel_size=3, padding=1)]

    # extraの(source6)に対する畳み込み層
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[5] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[5] * num_classes, kernel_size=3, padding=1)]
    
    return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)