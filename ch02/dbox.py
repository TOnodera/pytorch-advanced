from itertools import product
import numpy as np
import torch


class DBox(object):
    def __init__(self, cfg) -> None:
        super(DBox, self).__init__()

        # 画像サイズの300
        self.image_size = cfg['input_size']
        # 各sourceの特徴量マップのサイズ
        self.feature_maps = cfg['feature_maps']
        # sourceの個数
        self.num_priors = len(cfg['feature_maps'])
        # DBoxのピクセルサイズ
        self.steps = cfg['steps']
        # 小さい正方形のDBoxのピクセル
        self.min_sizes = cfg['min_sizes']
        # 大きい正方形のDBoxのピクセル
        self.max_sizes = cfg['max_sizes']
        # 長方形のアスペクト比
        self.aspect_ratios = cfg['aspect_ratios']

    def make_dbox_list(self):
        """
        DBoxを作成する
        """
        mean = []
        for k, f in enumerate(self.feature_maps):
            # fまでの数で2ペアの組み合わせをつくる
            for i,j in product(range(f), repeat=2):
                # 特徴量の画像サイズ
                # 300 / 'steps': [8, 16, 32, ...]
                f_k = self.image_size / self.steps[k]

                # DBoxの中心座標
                cx = (j+0.5) / f_k
                cy = (i+0.5) / f_k

                # アスペクト比1の小さいDBox [cx, cy, width, height]
                # 'min_size': [30,60, 111, 162, ...]
                s_k = self.min_sizes[k] / self.image_size
                mean += [cx, cy, s_k, s_k]

                # アスペクト比1の大きいDBox
                s_k_prime = np.sqrt(s_k*(self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # その他のアスペクト比のDBox
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*np.sqrt(ar), s_k/np.sqrt(ar)]
                    mean += [cx, cy, s_k/np.sqrt(ar), s_k*np.sqrt(ar)]
                    
        output = torch.Tensor(mean).view(-1, 4)
        output.clamp_(max=1, min=0)
        
        return output

