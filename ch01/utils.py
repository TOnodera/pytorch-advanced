import glob
import os.path as osp

def make_datapath_list(phase="train"):
    root_path = "./data/hymenoptera_data/"
    target_path = osp.join(root_path+phase+'/**/*.jpg')
    
    path_list = []
    
    for path in glob.glob(target_path):
        path_list.append(path)

    return path_list