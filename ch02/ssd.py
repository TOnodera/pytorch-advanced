from torch import nn
from l2norm import L2norm
import utils
from dbox import DBox

class SSD(nn.Module):
    def __init__(self, phase, cfg):
        super(SSD, self).__init__()

        self.phase = phase
        self.num_classes = cfg['num_classes']
        
        # SSDネットワークを作る
        self.vgg = utils.make_vgg()
        self.extras = utils.make_extras()
        self.L2Norm = L2norm()
        self.loc, self.conf = utils.make_loc_conf(cfg['num_classes'], cfg['bbox_aspect_num'])
        
        # DBox作成
        dbox = DBox(cfg)
        self.dbox_list = dbox.make_dbox_list()

        # 推論時はDetectを用意する
        if phase == 'inference':
            self.detect = Detect()
        
