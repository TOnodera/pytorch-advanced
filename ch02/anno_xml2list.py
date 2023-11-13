from typing import Any
import xml.etree.ElementTree as ET
import numpy as np

class Anno_xml2list(object):
    def __init__(self, classes) -> None:
        self.classes = classes
        
    def __call__(self,xml_path, width, height) -> Any:
        ret = []

        xml = ET.parse(xml_path).getroot()

        for obj in xml.iter('object'):
            # アノテーション見地でdifficultに設定されているものは除外
            difficult = int(obj.find('difficult').text)
            if difficult == 1:
                continue
            
            bndbox = []
            name = obj.find('name').text.lower().strip() # 物体名
            bbox = obj.find('bndbox') # バウンディングボックスの情報

            # アノテーションのxmin, ymin, xmax, ymaxを取得し0~1に規格化
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            
            for pt in (pts):
                # VOCは原点が(1,1)なので引き算して(0,0)にする
                cur_pixel = int(bbox.find(pt).text) - 1

                # 幅高さで規格化
                if pt == 'xmin' or pt == 'xmax':
                    cur_pixel /= width
                else:
                    cur_pixel /= height
                
                bndbox.append(cur_pixel)
                
            label_idx = self.classes.index(name)
            bndbox.append(label_idx)
            
            ret += [bndbox]
            
        return np.array(ret)
