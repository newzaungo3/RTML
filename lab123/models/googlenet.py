import torch
import torchvision
import torch.nn as nn
from .inception import Inception 

class GoogLeNet(nn.Module):
    '''
    GoogLeNet-like CNN

    Attributes
    ----------
    pre_layers : Sequential
        Initial convolutional layer
    a3 : Inceptionsg
        First inception block
    b3 : Inception
        Second inception block
    maxpool : MaxPool2d
        Pooling layer after second inception block
    a4 : Inception
        Third inception block
    b4 : Inception
        Fourth inception block
    c4 : Inception
        Fifth inception block
    d4 : Inception
        Sixth inception block
    e4 : Inception
        Seventh inception block
    a5 : Inception
        Eighth inception block
    b5 : Inception
        Ninth inception block
    avgpool : AvgPool2d
        Average pool layer after final inception block
    linear : Linear
        Fully connected layer
    '''

    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=7,padding=3,stride=2),
            nn.BatchNorm2d(64), 
            nn.MaxPool2d(3,stride=2,padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.MaxPool2d(3,stride=2,padding=1),
            nn.ReLU(True),
        )

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.linear = nn.Linear(1024, 10)

        self.aux1 = nn.Sequential(
            #1 convo 2 linear 
            nn.AvgPool2d(5,stride=3),
            nn.Conv2d(512,128,kernel_size=1,stride=1),
            nn.Flatten(start_dim=1),
            nn.Linear(2048,1024),
            nn.Dropout(0.7),
            nn.Linear(1024,10)
        )
        self.aux2 = nn.Sequential(
            #1 convo 2 linear 
            nn.AvgPool2d(5,stride=3),
            nn.Conv2d(528,128,kernel_size=1,stride=1),
            nn.Flatten(start_dim=1),
            nn.Linear(2048,1024),
            nn.Dropout(0.7),
            nn.Linear(1024,10)
        )

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)

        out2 = self.aux1(out)

        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)

        out3 = self.aux2(out)

        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out,out2,out3
