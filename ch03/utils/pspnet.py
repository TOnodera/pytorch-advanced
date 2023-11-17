from torch import nn
import torch.nn.functional as F
import torch

class PSPNet(nn.Module):
    def __init__(self, n_classes):
        super(PSPNet, self).__init__()

        # パラメータ設定
        block_config = [3, 4, 6, 3]
        img_size = 475
        img_size_8 = 60 # img_sizeの1/8に

        self.feature_conv = FeatureMap_convolution()
        self.feature_res_1 = ResidualBlockPSP(
            n_blocks=block_config[0], in_channels=128, mid_channels=64, out_channels=256, stride=1, dilation=1
        )
        self.feature_res_2 = ResidualBlockPSP(
            n_blocks=block_config[1], in_channels=256, mid_channels=128, out_channels=512, stride=2, dilation=1
        )
        self.feature_dilated_res_1 = ResidualBlockPSP(
            n_blocks=block_config[2], in_channels=512, mid_channels=256, out_channels=1024, stride=1, dilation=2
        )
        self.feature_dilated_res_2 = ResidualBlockPSP(
            n_blocks=block_config[3], in_channels=1024, mid_channels=512, out_channels=2048, stride=1, dilation=4
        )
        
        self.pyramid_pooling = PyramidPooling(in_channels=2048, pool_size=[6,3,2,1], height=img_size_8, width=img_size_8)
        self.decode_feature = DecodePSPFeature(height=img_size, width=img_size, n_classes=n_classes)
        self.aux=AuxiliaryPSPLayers(in_channels=1024, height=img_size, width=img_size,n_classes=n_classes)
        

    def forward(self, x):
        x = self.feature_conv(x)
        x = self.feature_res_1(x)
        x = self.feature_res_2(x)
        x = self.feature_dilated_res_1(x)
        output_aux = self.aux(x)
        x = self.feature_dilated_res_2(x)
        x = self.pyramid_pooling(x)
        output = self.decode_feature(x)
        return (output, output_aux)

class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
        super(conv2DBatchNormRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.Relu(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        outputs = self.relu(x)
        return outputs

class FeatureMap_convolution(nn.Module):
    def __init__(self):
        super(FeatureMap_convolution, self).__init__()
        
        # 1
        in_channels = 3
        out_channels = 64
        kernel_size = 3
        stride = 2
        padding = 1
        dilation = 1
        bias = False
        self.cbnr_1 = conv2DBatchNormRelu(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)
        
        # 2
        in_channels = 64 
        out_channels = 64
        kernel_size = 3
        stride = 1 
        padding = 1
        dilation = 1
        bias = False
        self.cbnr_2 = conv2DBatchNormRelu(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)
        
        # 3
        in_channels = 64 
        out_channels = 128 
        kernel_size = 3
        stride = 1 
        padding = 1
        dilation = 1
        bias = False
        self.cbnr_3 = conv2DBatchNormRelu(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        x = self.cbnr_1(x)
        x = self.cbnr_2(x)
        x = self.cbnr_3(x)
        outputs = self.maxpool(x)
        return outputs

class ResidualBlockPSP(nn.Sequential):
    def __init__(self, n_blocks, in_channels, mid_channels, out_channels, stride, dilation) -> None:
        super(ResidualBlockPSP, self).__init__()

        self.add_module(
            "block1",
            bottleNeckPSP(in_channels, mid_channels, out_channels, stride, dilation)
        )
        
        for i in range(n_blocks - 1):
            self.add_module(
                "block" + str(i+2),
                bottleNeckIdentifyPSP(
                    out_channels, mid_channels, stride, dilation
                )
            )
            
class conv2DBatchNorm:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias) -> None:
        super(conv2DBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.batchnorm = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        outputs = self.batchnorm(x)
        return outputs

class bottleNeckPSP(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride, dilation):
        super(conv2DBatchNorm, self).__init__()

        self.cbr_1 = conv2DBatchNormRelu(in_channels, mid_channels, kernel_size=1, stride=1, padding=0,dilation=1, bias=False)
        self.cbr_2 = conv2DBatchNormRelu(in_channels, mid_channels, kernel_size=3, stride=stride, padding=dilation,dilation=dilation, bias=False)
        self.cbr_3 = conv2DBatchNormRelu(in_channels, mid_channels, kernel_size=1, stride=1, padding=0,dilation=1, bias=False)
        
        # スキップ結合
        self.cb_residual = conv2DBatchNorm(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, dilation=1, bias=False)

    def forward(self, x):
        conv = self.cbr_3(self.cbr_2(self.cb1(x)))
        residual = self.cb_residual(x)
        return self.relu(conv + residual)

class bottleNeckIdentifyPSP(nn.Module):
    def __init__(self, in_channels, mid_channels, stride, dilation):
        super(bottleNeckIdentifyPSP, self).__init__()
        
        self.cbr_1 = conv2DBatchNormRelu(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.cbr_2 = conv2DBatchNormRelu(mid_channels, mid_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.cbr_3 = conv2DBatchNormRelu(mid_channels, in_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        conv = self.cbr_3(self.cbr_2(self.cbr_1(x)))
        residual = x
        return self.relu(conv + residual)
        

class PyramidPooling(nn.Module):
    def __init__(self, in_channels, pool_size, height, width):
        super(PyramidPooling, self).__init__()
        self.height = height
        self.widht = width
        
        out_channels = int(in_channels/len(pool_size))
        
        # pool_size = [6, 3, 2, 1]
        self.avpool_1 = nn.AdaptiveAvgPool2d(output_size=pool_size[0])
        self.cbr_1 = conv2DBatchNormRelu(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False
        )

        self.avpool_2 = nn.AdaptiveAvgPool2d(output_size=pool_size[1])
        self.cbr_2 = conv2DBatchNormRelu(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False
        )

        self.avpool_3 = nn.AdaptiveAvgPool2d(output_size=pool_size[2])
        self.cbr_3 = conv2DBatchNormRelu(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False
        )

        self.avpool_4 = nn.AdaptiveAvgPool2d(output_size=pool_size[3])
        self.cbr_4 = conv2DBatchNormRelu(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False
        )
        

    def forward(self, x):
        out1 = self.cbr_1(self.avpool_1(x))
        out1 = F.interpolate(out1, size=(self.height, self.widht), mode="bilinear",align_corners=True)
        
        out2 = self.cbr_2(self.avpool_2(x))
        out2 = F.interpolate(out2, size=(self.height, self.widht), mode="bilinear",align_corners=True)
        
        out3 = self.cbr_3(self.avpool_3(x))
        out3 = F.interpolate(out3, size=(self.height, self.widht), mode="bilinear",align_corners=True)

        out4 = self.cbr_4(self.avpool_4(x))
        out4 = F.interpolate(out4, size=(self.height, self.widht), mode="bilinear",align_corners=True)
        
        # 最終的に結合させる
        output = torch.cat([x, out1, out2, out3, out4], dim=1)
        
        return output