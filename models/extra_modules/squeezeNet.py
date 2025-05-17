import torch
import torch.nn as nn
import torch.nn.init as init


# __all__ = ['SqueezeNet', 'squeezenet1_0', 'squeezenet1_1']


# 定义Fire模块的属性
class Fire(nn.Module):
    # __init__定义参数属性初始化，接着使用super(Fire, self).__init__()调用基类初始化函数对基类进行初始化
    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        # inplanes:输入通道 squeeze_planes:输出通道 expand1x1_planes：1x1卷积层 expand3x3_planes：3x3卷积层
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        # inplace=True表示可以在原始的输入上直接操作，不会再为输出分配额外内存，但同时会破坏原始的输入
        self.expand1x1_activation = nn.ReLU(inplace=True)
        # 为了使1x1和3x3的filter输出的结果又相同的尺寸，在expand modules中，给3x3的filter的原始输入添加一个像素的边界（zero-padding）.
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        # squeeze 和 expand layers中都是用ReLU作为激活函数
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        # 将输入x经过squeeze layer进行卷机操作，再经过squeez_activation()进行激活
        x = self.squeeze_activation(self.squeeze(x))
        # 将输出分别送到expand1x1和expand3x3中进行卷积核激活
        # 最后使用torch.cat()将两个相同维度的张量连接在一起，这里dim=1，代表按列连接，最终得到若干行
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

# version=1.0和version=1.1两个SqueezeNet版本
# version=1.0只有AlexNet的1/50的参数，而1.1在1.0的基础上进一步压缩，计算了降低为1.0的40%左右，主要体现在(1)第一层卷积核7x7更改为3x3 (2)将max_pool层提前了，减少了计算量
class SqueezeNet(nn.Module):
    # num_classes:分类的类别个数
    def __init__(self, version='squeezenet1_0'):
        super(SqueezeNet, self).__init__()
        if version == 'squeezenet1_0':
            # self.features:定义了主要的网络层，nn.Sequential()是PyTorch中的序列容器
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=3),
                nn.ReLU(inplace=True),
                # ceil_mode=True会对池化结果进行向上取整而不是向下取整
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            )
        elif version == 'squeezenet1_1':
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            )
        else:
            # FIXME: Is this needed? SqueezeNet should only be called from the
            # FIXME: squeezenet1_x() functions
            # FIXME: This checking is not done for the other models
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1_0 or 1_1 expected".format(version=version))

        self._initialize_weights()

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        _, _, h, w = x.size()
        scale = [4, 8, 16, 32]
        out = []
        shape = [x.shape[2] // i for i in scale]

        for i, model in enumerate(self.features):
            x1 = x
            x = model(x)
            if x1.shape[2] != x.shape[2] and x1.shape[2] in shape:
                out.append(x1)

        out.append(x)
        return out


# squeezenet1_0', 'squeezenet1_1'
if __name__ == "__main__":
    model = SqueezeNet('squeezenet1_1')
    y = model(torch.zeros(1, 3, 64, 64))
    print(y[0].shape)
    print(y[1].shape)
    print(y[2].shape)
    print(y[3].shape)
    print(len(y))
