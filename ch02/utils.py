import os.path as osp
import torch

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
    
    # targetsはアノテーションデータの正解であるgtのリストです。
    # リストのサイズはミニバッチのサイズです。
    # リストのtargetsの要素は[n, 5]となっています。
    # nは画像ごとに異なり、画像内の物体数となっています。
    # 5は[xmin, ymin, xmax, ymin, class_index]です。
    
    return imgs, targets
