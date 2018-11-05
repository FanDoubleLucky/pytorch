# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 12:49:49 2018

@author: FYZ
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.functional as F

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet,self).__init__
        self.conv1 = nn.Conv2d(3,64,3)
        self.layer1 = self.make_layer(ResidualBlock)
        self.layer2 = self.make_layer(ResidualBlock)
        self.layer3 = self.make_layer(ResidualBlock)
        self.layer4 = self.make_layer(ResidualBlock)
        self.fc1  = nn.Linear(512,10)
        
    def make_layer(self,Block,in_channel,out_channel):
        layer = []
        
        return nn.Sequential(*layer)
    
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x,4)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self,inchannel,outchannel,stride=1):
        super(ResidualBlock,self).__init__()
        
        
    def forward(self,x):
        