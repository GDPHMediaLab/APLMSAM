import torch
import torch.nn as nn
import torch.nn.functional as F


class DownsampleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsampleConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):

        x = self.conv(x)


        x = self.norm(x)

        x = self.gelu(x)

        return x



class ConvUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvUpsample, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x_enc, x_dec):
        x = self.upconv(x_enc)
        x = torch.cat((x, x_dec), dim=1)
        return self.conv(x)

class ConvUpsample1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvUpsample1, self).__init__()

        self.upconv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x_enc, x_dec):
        x = self.upconv1(x_enc)
        x = torch.cat((x, x_dec), dim=1)
        return self.conv(x)

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class CNN_APB(nn.Module):
    def __init__(self):
        super(CNN_APB, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=160, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(160),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=320, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(320),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1, stride=1)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=320, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(320),
            nn.ReLU()
        )


        self.downsample1 = DownsampleConv(in_channels=128 * 2,
                                     out_channels=128)
        self.downsample2 = DownsampleConv(in_channels=320, out_channels=160)
        self.downsample3 = DownsampleConv(in_channels=640, out_channels=320)
        self.downsample4 = DownsampleConv(in_channels=640, out_channels=320)

        #Auto_prompt_block
        self.decoder4 = ConvUpsample1(320, 320)
        self.decoder3 = ConvUpsample1(320, 160)
        self.decoder2 = ConvUpsample(160, 128)
        self.decoder1 = ConvUpsample(128, 3)
        self.decoder0 = torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1)


    def forward(self, x_c,x0,x1,x2,x3):

        x_0 = self.conv1(x_c)  # (1, 128, 128, 128)
        x_0 = torch.cat((x_0,x0),dim=1)
        x_0 = self.downsample1(x_0)

        x_1 = self.conv2(x_0)  # (1,160, 64, 64)

        x_1 = torch.cat((x_1, x1), dim=1)
        x_1 = self.downsample2(x_1)

        x_2 = self.conv3(x_1)  # (1, 320, 64, 64)
        x_2 = torch.cat((x_2, x2), dim=1)
        x_2 = self.downsample3(x_2)


        x_3 = self.conv4(x_2)  # (1, 320, 64, 64)
        x_3 = torch.cat((x_3, x3), dim=1)
        x_3 = self.downsample4(x_3)



        x = self.decoder4(x_3, x_2)

        x = self.decoder3(x, x_1)

        x = self.decoder2(x, x_0)


        x = self.decoder1(x, x_c)


        x = self.decoder0(x)



        return x




