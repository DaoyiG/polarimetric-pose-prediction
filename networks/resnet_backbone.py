import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck
from mmcv.cnn import normal_init, constant_init


# Specification
resnet_spec = {
    18: (BasicBlock, [2, 2, 2, 2], [64, 64, 128, 256, 512], "resnet18"),
    34: (BasicBlock, [3, 4, 6, 3], [64, 64, 128, 256, 512], "resnet34"),
    50: (Bottleneck, [3, 4, 6, 3], [64, 256, 512, 1024, 2048], "resnet50"),
    101: (Bottleneck, [3, 4, 23, 3], [64, 256, 512, 1024, 2048], "resnet101"),
    152: (Bottleneck, [3, 8, 36, 3], [64, 256, 512, 1024, 2048], "resnet152"),
}


class ResNetBackboneNet(nn.Module):
    def __init__(self, block, layers, in_channel=3, use_skips=False):
        super(ResNetBackboneNet, self).__init__()
        self.use_skips = use_skips
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x): 
        self.features = []

        x = self.conv1(x) 
        x = self.bn1(x)
        x = self.relu(x)
        x_low_feature = self.maxpool(x) 
        x_f64 = self.layer1(x_low_feature)  
        x_f32 = self.layer2(x_f64) 
        x_f16 = self.layer3(x_f32)  
        x_high_feature = self.layer4(x_f16)
        if self.use_skips:
            self.features.append(x_f64)
            self.features.append(x_f32)
            self.features.append(x_f16) 
            self.features.append(x_high_feature)
            return self.features
        else:
            self.features.append(x_high_feature)
            return self.features

