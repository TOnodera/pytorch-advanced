import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import models, transforms
from base_transform import BaseTransform
from ilsvrc_predictor import ILSVRCPredictor

# ラベル情報のロード
ILSVRC_class_index = json.load(open("./data/imagenet_class_index.json", "r"))
predictor = ILSVRCPredictor(ILSVRC_class_index) 

image_file_path = "./data/goldenretriever-3724972_640.jpg"
img = Image.open(image_file_path)

resize = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
transform = BaseTransform(resize, mean, std)
img_transformed = transform(img)
inputs = img_transformed.unsqueeze_(0)

# VGG-16モデルをロード
use_pretrained = True
net = models.vgg16(pretrained=use_pretrained)
# # 推論モードに設定
net.eval()

out = net(inputs)
result = predictor.predict_max(out)

print("予測結果: ", result)

