"""
Code based on https://github.com/usuyama/pytorch-unet/blob/master/pytorch_unet.py
"""
import torch
import torch.nn.functional as F
from torch import nn


class ConvRelu2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # we use a padding of 1 so the image does not get smaller after every conv layer
        # this is different than the architecture described in the paper, but makes the code much simpler,
        # because the images do not have to be cropped from the encoder output to the decoder input.
        # also, this way the output image has the same size as the input image
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)

        x = self.batch_norm2(x)
        x = F.relu(x, inplace=True)
        return x


class UNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.enc_conv1 = ConvRelu2(in_channels, 64)
        self.enc_conv2 = ConvRelu2(64, 128)
        self.enc_conv3 = ConvRelu2(128, 256)
        self.enc_conv4 = ConvRelu2(256, 512)
        self.enc_conv5 = ConvRelu2(512, 1024)

        self.dec_conv1 = ConvRelu2(512 + 1024, 512)  # conv4 + conv5
        self.dec_conv2 = ConvRelu2(256 + 512, 256)  # conv3 + conv4
        self.dec_conv3 = ConvRelu2(128 + 256, 128)  # conv2 + conv3
        self.dec_conv4 = ConvRelu2(64 + 128, 64)  # conv1 + conv2

        self.output_conv = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1, padding=0)

    def upsample(self, x):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        # encoder
        enc_conv1_res = self.enc_conv1(x)
        x = F.max_pool2d(enc_conv1_res, 2)

        enc_conv2_res = self.enc_conv2(x)
        x = F.max_pool2d(enc_conv2_res, 2)

        enc_conv3_res = self.enc_conv3(x)
        x = F.max_pool2d(enc_conv3_res, 2)

        enc_conv4_res = self.enc_conv4(x)
        x = F.max_pool2d(enc_conv4_res, 2)

        x = self.enc_conv5(x)

        # decoder
        x = self.upsample(x)
        x = torch.cat((x, enc_conv4_res), dim=1)
        x = self.dec_conv1(x)

        x = self.upsample(x)
        x = torch.cat((x, enc_conv3_res), dim=1)
        x = self.dec_conv2(x)

        x = self.upsample(x)
        x = torch.cat((x, enc_conv2_res), dim=1)
        x = self.dec_conv3(x)

        x = self.upsample(x)
        x = torch.cat((x, enc_conv1_res), dim=1)
        x = self.dec_conv4(x)

        x = self.output_conv(x).squeeze(1)
        x = torch.sigmoid(x)
        return x
