import torch
from torch import nn
from unet_model import UNet

class Denoiser(nn.Module):
    def __init__(self,device,\
                 in_features,
                 out_features):
        super(Denoiser,self).__init__()
        self.device = device
        self.in_features = in_features
        
        self.model = UNet(in_features,out_features)
        
    def forward(self,x):
        y = self.model(x)
        return 2*(torch.sigmoid(y) - 0.5)