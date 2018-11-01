# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 23:48:47 2018

@author: FYZ
"""

import torch
from first_classifier import Net
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

input = torch.ones(3,3,32,32)+3
out = net(input)
criterion = nn.CrossEntropyLoss()
target = Variable(torch.ones(3,10))
loss = criterion(target,out)
optimizer.zero_grad()
loss.backward()