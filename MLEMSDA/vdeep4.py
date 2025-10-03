
import torch.nn as nn
from torch.nn import functional as F

class deep(nn.Module):
    def __init__(self):
        super(deep, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(10, 1), stride=(1, 1))
        self.conv1_1 = nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 11), stride=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.max1 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1), padding=0, dilation=1, ceil_mode=False)

        self.conv2 = nn.Conv2d(in_channels=25, out_channels=50, kernel_size=(10, 1), stride=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.max2 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1), padding=0, dilation=1, ceil_mode=False)

        self.conv3 = nn.Conv2d(in_channels=50, out_channels=100, kernel_size=(10, 1), stride=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(100,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.max3 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1), padding=0, dilation=1, ceil_mode=False)

        self.conv4 = nn.Conv2d(in_channels=100, out_channels=200, kernel_size=(10, 1), stride=(1, 1), bias=False)
        self.bn4 = nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.max4 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1), padding=0, dilation=1, ceil_mode=False)
        self.conv_classifier = nn.Conv2d(in_channels=200, out_channels=2, kernel_size=(20, 1), stride=(1, 1))
    def forward(self, x):  # (64, 1, 1000, 11)
        out = self.conv1(x)  # (64 , 25, 991, 11)
        out = self.conv1_1(out)  # (64 , 25, 991, 1)
        out = F.elu(self.bn1(out))
        out = F.dropout(self.max1(out), p=0.5, training=self.training)
        out = self.conv2(out)#(64, 50, 321, 1)
        out = F.elu(self.bn2(out))
        out = F.dropout(self.max2(out), p=0.5, training=self.training)
        out = self.conv3(out)#[64, 100, 98, 1]
        out = F.elu(self.bn3(out))
        out = F.dropout(self.max3(out), p=0.5, training=self.training)
        out = self.conv4(out)#[64, 200, 23, 1]
        out = F.elu(self.bn4(out))
        out = self.max4(out)
        out = self.conv_classifier(out)


        return out


