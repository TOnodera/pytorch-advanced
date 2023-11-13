from torch import nn
import torch

class L2norm(nn.Module):
    def __init__(self, input_channels=512, scale=20):
        super(L2norm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(input_channels))
        self.scale = scale
        self.reset_parameters()
        self.eps = 1e-10
        
    def reset_parameters(self):
        """
        結合パラメータの大きさをscaleで初期化する
        """
        nn.init.constant_(self.weight, self.scale)
        
    def forward(self, x):
        """
        38x38の特徴量に対して、512チャネルにわたって2乗和のルートをもとめた38x38
        の値を使用して、核とr区長量を正規化してから軽巣を掛け算する層
        """
        
        # 奥行（チャネル)方向のL2ノルムを計算して平方根をとる
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        print(f"norm: {norm}")
        x = torch.div(x, norm)
        
        # unsqeezeは指定したサイズ１の次元を追加する
        weights = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)
        out = weights * x
        
        return out
        

