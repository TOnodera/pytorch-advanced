import numpy as np
class ILSVRCPredictor:
    """
    ILSVRCデータに対する戻るの出力からラベルを求める
    """
    def __init__(self, class_index) -> None:
        self.class_index = class_index

    def predict_max(self, out):
        maxid = np.argmax(out.detach().numpy()) 
        predicted_label_name = self.class_index[str(maxid)][1]
        return predicted_label_name