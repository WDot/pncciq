import torch
from torch import nn
import torch.nn.functional as F

class ConvMaxPool(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=23, stride=1):
        super(ConvMaxPool, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=0)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class ConvAvgPool(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=23, stride=1):
        super(ConvAvgPool, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=0)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)  # Adaptive pooling to a fixed size

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x.squeeze(-1)  # Flatten last dimension

class CNN(nn.Module):
    def __init__(self, in_features, num_classes):
        super(CNN, self).__init__()
        self.conv1 = ConvMaxPool(in_features, 16)
        self.conv2 = ConvMaxPool(16, 24)
        self.conv3 = ConvMaxPool(24, 32)
        self.conv4 = ConvMaxPool(32, 48)
        self.conv5 = ConvMaxPool(48, 64)
        self.conv6 = ConvAvgPool(64, 96)
        self.do = nn.Dropout(0.5)
        self.fc = nn.Linear(96, num_classes)  # Fully connected layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.fc(self.do(x))
        return x  # Raw logits (Softmax should be applied outside for stability)
    
class Predictor(nn.Module):
    def __init__(self,
                 in_features,\
                 num_classes,\
                 scale):
        torch.set_default_dtype(torch.float32)
        super(Predictor,self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.scale = scale

        self.model = CNN(self.in_features,1)
        
    def forward(self,x):
        y = self.model(x)
        #classes = y[:,:self.num_classes]
        tdelayestimate = 2*(torch.sigmoid(y) - 0.5)*self.scale


        return tdelayestimate