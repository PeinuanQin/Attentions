import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes
                     , out_planes
                     , kernel_size=3
                     , stride=stride
                     , padding=1
                     , bias=False)


class BasicBlock(nn.Module):

    def __init__(self
                 , inplanes
                 , planes
                 , stride=1):
        super(BasicBlock, self).__init__()
        self.inplanes = inplanes
        self.outplanes = planes

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x

        '''将一个 block 中的所有 layer 生成一个 block attention'''

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + residual
        out = self.relu(out)
        return out


class BasicGroup(nn.Module):
    def __init__(self
                 , inplanes
                 , planes
                 , n_blocks):
        super(BasicGroup,self).__init__()
        self.n_blocks = n_blocks
        self.blocks = nn.Sequential(*[BasicBlock(inplanes
                                                 ,planes) for _ in range(n_blocks)])
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        out = x
        for i in range(self.n_blocks):
            '''这里的 block attention 是没有 reduction 的'''
            out = self.blocks[i](out)
        return out


class DownSampleBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=2):
        super(DownSampleBlock, self).__init__()
        self.inplanes = inplanes
        self.outplanes = planes
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.stride = stride
        self.downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes)
        )


    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        '''如果这个 block 最后 downsample 了，那么 residual 要保持一致'''
        residual = self.downsample(x)
        out = out + residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, depth, num_classes, fusion_module, channel=[16, 32, 64]):
        super(ResNet, self).__init__()
        self.channel = channel
        self.depth = depth
        self.inplanes = self.channel[0]
        self.fusion_module = fusion_module
        n = int((depth - 2) / 6)
        block = BasicBlock
        downsample_block = DownSampleBlock

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.group1 = BasicGroup(inplanes=self.inplanes
                                 ,planes=self.channel[0]
                                 ,n_blocks=n)
        self.block_downsample2 = downsample_block(inplanes=self.channel[0]
                                                  ,planes=self.channel[1])
        '''进入 group2 的特征维度是 o: (128,32,16,16)'''
        self.group2 = BasicGroup(inplanes=self.channel[1]
                                 ,planes=self.channel[1]
                                 ,n_blocks=n-1)

        self.block_downsample3 = downsample_block(self.channel[1]
                                                  ,planes=self.channel[2])
        self.group3 = BasicGroup(inplanes=self.channel[2]
                                 ,planes=self.channel[2]
                                 ,n_blocks=n-1
                                 )

        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(self.channel[2], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):

        o = self.conv1(x)
        o = self.bn1(o)
        o = self.relu(o)

        o = self.group1(o)

        o = self.block_downsample2(o)
        o = self.group2(o)

        o = self.block_downsample3(o)
        o = self.group3(o)

        out = self.avgpool(o)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

