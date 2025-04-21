import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    """
    ResNet的瓶颈块结构，用于ResNet50/101/152
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet101(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3, 4, 23, 3], num_classes=2, dropout=0.5, init_weights=True):
        super(ResNet101, self).__init__()
        self.in_channels = 64
        self.dropout_rate = dropout
        
        # 初始卷积层和池化层
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 四个残差层
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # 全局平均池化和分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0 else None
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # 初始化权重
        if init_weights:
            self._initialize_weights()
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        
        layers = []
        # 第一个block可能需要downsample
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        
        # 更新in_channels
        self.in_channels = out_channels * block.expansion
        
        # 添加剩余的blocks
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        if self.dropout is not None:
            x = self.dropout(x)
            
        x = self.fc(x)
        
        return x


def resnet101(num_classes=2, dropout=0.5, pretrained=False, **kwargs):
    """构建ResNet101模型
    
    Args:
        num_classes (int): 分类类别数
        dropout (float): Dropout比率
        pretrained (bool): 是否使用预训练模型
        
    Returns:
        ResNet101模型
    """
    model = ResNet101(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, dropout=dropout, **kwargs)
    
    if pretrained:
        # 如果有可用的预训练模型，此处可以加载
        # model.load_state_dict(torch.load('path/to/pretrained/model'))
        print("预训练模型不可用或已禁用")
    
    return model
