import matplotlib.pyplot as plt
from generator import Generator
import torch

G = Generator(z_dim=20, image_size=64) 

# 入力する乱数
input_z = torch.rand(1, 20)
# (1, 20, 1, 1)に変形
input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)

# 偽画像を出力
fake_images = G(input_z)

img_transformed = fake_images[0][0].detach().numpy()
plt.imshow(img_transformed, 'gray')
plt.show()