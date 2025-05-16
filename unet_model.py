""" Full assembly of the parts to form the complete network """

from unet_parts import DoubleConv,Down,Up,OutConv

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes,bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.divisor = 4
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64//self.divisor))
        self.down1 = (Down(64//self.divisor, 128//self.divisor))
        self.down2 = (Down(128//self.divisor, 256//self.divisor))
        self.down3 = (Down(256//self.divisor, 512//self.divisor))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512//self.divisor, 1024//self.divisor // factor))
        self.up1 = (Up(1024//self.divisor, 512//self.divisor // factor, bilinear))
        self.up2 = (Up(512//self.divisor, 256//self.divisor // factor, bilinear))
        self.up3 = (Up(256//self.divisor, 128//self.divisor // factor, bilinear))
        self.up4 = (Up(128//self.divisor, 64//self.divisor, bilinear))
        self.outc = (OutConv(64//self.divisor, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        fv = torch.squeeze(F.max_pool1d(x5,kernel_size=x5.size()[2:]),2) #Global max pool
        x,_ = self.up1(x5, x4)
        x,_ = self.up2(x, x3)
        x,_ = self.up3(x, x2)
        x,fm = self.up4(x, x1)
        logits = self.outc(x)
        return logits