from torchvision import transforms


class BaseTransform:
    """
    画像サイズをリサイズし、色を標準化する

    Attribute:
    ----------
    resize: int
        リサイズ先の画像の大きさ
    mean: (R, G, B)
        各色チャネルの平均値
    std: (R, G, B)
        各色チャネルの標準偏差
    """
    def __init__(self, resize, mean, std) -> None:
        self.base_transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
    def __call__(self, img):
        return self.base_transform(img)