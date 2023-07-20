import torch
import numpy
import torch.nn as nn


def weights_init_normal(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif class_name.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class UNetDown(nn.Module):
    """
        UNetDown for image process
        input:x
        output:encoded x
        structure:Conv2d-->Norm(if chosen)-->LeakRelu-->Dropout(if chosen)
    """
    def __init__(self, in_size, out_size, kernel_size=4, stride=2, padding=1, normalize=True, leaky=0.2, dropout=0.5):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)]

        if normalize:   # 是否归一化
            layers.append(nn.InstanceNorm2d(out_size))

        # layers.append(nn.LeakyReLU(leaky))    # 激活函数层
        layers.append(nn.Tanh())    # 激活函数层


        if dropout:   # 是否Dropout
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """
        UNetUp for image process
        input:x,skip_input
        output:model(x) cat skip_input
        structure:Conv2d-->Norm-->Relu-->Dropout(if chosen)
    """
    def __init__(self, in_size, out_size, kernel_size=4, stride=2, padding=1, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [nn.ConvTranspose2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                  nn.InstanceNorm2d(out_size),
                  nn.Tanh()]
                #   nn.ReLU(inplace=True)]
        if dropout:  # 是否Dropout
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        return torch.cat((x, skip_input), 1)


class UNet(nn.Module):
    """
        Network from image to image
        Input(Noisy Image N)--UNet-->Purified Image I
        ----------------------------------------------------------------------------
        Input:
        :param N:4D torch.tensor:(batch_size * RGB * Width * Height)
        Width and Height are 256 and 128
        ----------------------------------------------------------------------------
        Output:
        :return I:4D torch.tensor:(batch_size * RGB * Width * Height)
    """
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()
        # 定义encoder
        self.down1 = UNetDown(in_channels, 4, normalize=False)
        self.down2 = UNetDown(4, 8)
        self.down3 = UNetDown(8, 16)
        self.down4 = UNetDown(16, 32, dropout=0.5)
        self.down5 = UNetDown(32, 64, dropout=0.5)
        self.down6 = UNetDown(64, 128, dropout=0.5)
        self.down7 = UNetDown(128, 128, normalize=False, dropout=0.5)

        # 定义decoder
        self.up1 = UNetUp(128, 128, dropout=0.5)
        self.up2 = UNetUp(256, 64, dropout=0.5)
        self.up3 = UNetUp(128, 32, dropout=0.5)
        self.up4 = UNetUp(64, 16)
        self.up5 = UNetUp(32, 8)
        self.up6 = UNetUp(16, 4)

        # 定义输出
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(8, out_channels, 4, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)

        u1 = self.up1(d7, d6)
        u2 = self.up2(u1, d5)
        u3 = self.up3(u2, d4)
        u4 = self.up4(u3, d3)
        u5 = self.up5(u4, d2)
        u6 = self.up6(u5, d1)

        result = self.final(u6)

        return result


class ForwardRemover(nn.Module):
    """
        Module from noisy image to image
        Input(Noisy Image N)--UNet-->Purified Image I
        ----------------------------------------------------------------------------
        Input:
        :param N:4D torch.tensor:( batch_size * RGB * Width * Height )
        Width and Height are 256 and 128 or uncertainy
        ----------------------------------------------------------------------------
        Output:
        :return I:4D torch.tensor:(batch_size * RGB * Width * Height)
    """
    def __init__(self, in_channels=3, out_channels=3, scale=1):
        super(ForwardRemover, self).__init__()
        # 定义encoder
        self.model = UNet(in_channels=in_channels, out_channels=out_channels)
        self.scale = scale

    def forward(self, x):
        x = x/self.scale
        x = self.model(x)
        x = x*self.scale
        x = nn.Sigmoid()(x)
        return x


class ResRemover(nn.Module):
    """
        Module from noisy image to noise then obtaining noise-free image
        Input(Noisy Image N)--UNet-->noise
        Output(Noise-free Image I) = Input+noise
        ----------------------------------------------------------------------------
        Input:
        :param N:4D torch.tensor:(batch_size * RGB * Width * Height)
        Width and Height are 256 and 128 or uncertainy
        ----------------------------------------------------------------------------
        Output:
        :return I:4D torch.tensor:(batch_size * RGB * Width * Height)
    """
    def __init__(self, in_channels=3, out_channels=3, scale=1):
        super(ResRemover, self).__init__()
        # 定义encoder
        self.model = UNet(in_channels=in_channels, out_channels=out_channels)
        self.scale = scale


    def forward(self, x):
        x = x/self.scale
        noise = self.model(x)
        x = x+noise# Plus is equal to minus? For tanh is used
        x = x*self.scale
        x = nn.Sigmoid()(x)
        return x