from torch import nn
from l2norm import L2norm
import utils
from dbox import DBox
from detect import Detect
import torch.nn.functional as F
import torch

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
        
    def forward(self, x):
        sources = list()
        loc = list()
        conf = list()

        # conv4_3まで計算する
        for k in range(23):
            x = self.vgg[k](x)

        # conv4_3の出力をL2normに通してsource1を作成する
        source1 = self.L2Norm(x)
        sources.append(source1)

        # vggを最後まで計算してsource2を作成する
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)

        sources.append(x) # source2
        
        # extrasのconvとReluを計算
        # source3~6を作成する
        for k, v, in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            # conv -> Relu -> conv -> Reluしたらsourcesに入れる
            if k % 2 == 1:
                sources.append(x)
                
        # locとconfをまとめて処理する
        for (x, l, c) in zip(sources, self.loc, self.conf):
            # permuteで順番入れ替え
            loc.append(l(x).permute(0, 2, 3, 1).contiguos())
            conf.append(c(x).permute(0, 2, 3, 1).contiguos())

        # 形を変形 
        loc = torch.cat([o.view(o.size(o), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(o), -1) for o in conf], 1)
        
        # locの形はtorcy.Size([batch_num, 8732,4])
        # confの形はtorcy.Size([batch_num, 8732,21])
        # になる
        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes)
        
        output = (loc, conf, self.dbox_list)
        
        if self.phase == 'inference': # 推論時
            return self.detect(output[0], output[1], output[2])
        else: # 学習時
            return output

            