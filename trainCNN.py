# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 23:55:33 2018

@author: FYZ
"""

import torch
import torch.nn as nn
from firstCNN import Net
from torch.autograd import Variable


net = Net()
input = torch.randn(1,1,32,32)
output = net(input)
print(output)
target = Variable(torch.ones(1,10))
criterion = nn.MSELoss()
loss = criterion(output,target)
print(loss)