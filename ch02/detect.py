from torch.autograd import Function
from torch import nn
import torch
import utils

class Detect(Function):
    def __init__(self, conf_thresh=0.01, top_k=200, nms_thresh=0.45):
        self.softmax = nn.Softmax(dim=-1)
        # confがconf_thresh=0.01より高いDBoxのみを扱う
        self.conf_thresh = conf_thresh
        self.top_k = top_k
        # nm_supressionでIOUがnms_thresh=0.45より大きい場合は同一物体へのBBoxとみなす
        self.nms_thresh = nms_thresh 
        
    def forward(self, loc_data, conf_data, dbox_list):
        """
        順伝搬の計算を行う

        Parameters
        ---------- 
        loc_data: [batch_num, 8732, 4]
            オフセット情報
        conf_data: [batch_num, 8732, 1]
            検出の確信度
        dbox_list: [8732, 4]
            DBoxの情報
        
        Returns
        ---------- 
        output: tourch.Size(batch_num, 21, 200, 5)
            (batch_num, クラス情報,確信度上位200, BBox情報)
        """
        
        num_batch = loc_data.size(0) # バッチサイズ
        num_dbox = loc_data.size(1) # DBoxの数
        num_classes = conf_data.size(2) #　分類するクラスの数(21)

        # softmaxで正規化
        conf_data = self.softmax(conf_data)

        # 出力の型を0埋めで作成
        output = torch.zeros(num_batch, num_classes, self.top_k, 5)
        
        # conf_dataを[batch_num, 8732, num_classes]から[batch_num, num_classes, 8732]に変換
        conf_preds = conf_data.transpose(2, 1)

        # ミニバッチごとのループ
        for i in range(num_batch):

            # locとDBoxからBBoxを求める
            decoded_boxes = utils.decode(loc_data[i], dbox_list)

            # confのコピーを作成
            conf_scores = conf_preds[i].clone()
            
            # 画像クラスごとのループ
            for cl in range(1, num_classes):

                # confの閾値を超えたBBoxを取り出す
                # 閾値を超えたものが１にそれ以外が０になる
                # cmaskは一次元配列になる 
                c_mask = conf_scores[i].gt(self.conf_thresh)
                
                scores = conf_scores[cl][c_mask]
                
                if scores.nelement() == 0:
                    continue

                l_mask = c_mask.unsqeeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)

                # Non-Maximum Suppresstionを実行してかぶってるBBoxを取り除く
                ids, count = utils.nm_suppression(boxes, scores, self.nms_thresh, self.top_k)
                
                # 処理結果を連結してから格納（むずいな）
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqeeze(1), boxes[ids[:count]]), 1)

        return output        
        

                