import torch
import torch.nn as nn
import math

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class TinyResNetBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, kernel_size=3):
        super(TinyResNetBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class TinyResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, input_channels=3):
        super(TinyResNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.inplanes = 16
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride*2),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


class ResnetSiamese(nn.Module):

    def __init__(self, resnet_layers, resnet_outsize, fc_layers):
        super(ResnetSiamese, self).__init__()

        self.tinyresnet = TinyResNet(TinyResNetBlock, resnet_layers, num_classes=resnet_outsize)
        self.fc1 = nn.Linear(resnet_outsize * 4, fc_layers[0])
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(fc_layers[0], fc_layers[1])
        self.softmax = nn.Softmax(dim=0)

    def forward_once(self, x):

        x = self.tinyresnet(x)

        return x


    def forward(self, x, y):

        # Pass examples through siamese resnet
        f_x = self.forward_once(x)
        f_y = self.forward_once(y)

        # Concatenate outputs
        squared_diff = (f_x - f_y)**2
        hadamard = (f_x * f_y)
        x = torch.cat((f_x,f_y,squared_diff,hadamard),1)

        # Pass through fully connected layers
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x
