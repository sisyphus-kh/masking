import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class FireModule(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(FireModule, self).__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)

        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)

        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False, batchnorm=True, dropout_prob=0.5):
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
        ]
        layers += [
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout_prob))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    in_channels = 4

    def prepare_image(self, x):
        return x

    def __init__(self, out_channels, dropout_prob=0.5):
        super(UNet, self).__init__()

        # Replace first encoding layer with SqueezeNet-like block
        self.enc1 = nn.Sequential(
            FireModule(self.in_channels, 8, 16, 16),  # Output: 32 channels (16+16)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)  # 256x256

        self.enc2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)  # 128x128

        self.enc3 = ConvBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2)  # 64x64

        self.enc4 = ConvBlock(128, 256)
        self.pool4 = nn.MaxPool2d(2)  # 32x32

        self.bottleneck = ConvBlock(256, 512)

        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(512, 256, batchnorm=False)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(256, 128, batchnorm=False)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128, 64, batchnorm=False)

        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(64, 32, dropout=True, batchnorm=False, dropout_prob=dropout_prob)

        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.prepare_image(x)

        enc1 = self.enc1(x)
        enc1_pool = self.pool1(enc1)

        enc2 = self.enc2(enc1_pool)
        enc2_pool = self.pool2(enc2)

        enc3 = self.enc3(enc2_pool)
        enc3_pool = self.pool3(enc3)

        enc4 = self.enc4(enc3_pool)
        enc4_pool = self.pool4(enc4)

        bottleneck = self.bottleneck(enc4_pool)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        dec0 = self.out_conv(dec1)
        out = self.sigmoid(dec0)
        return out
