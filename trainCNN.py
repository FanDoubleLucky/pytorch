# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 23:55:33 2018

@author: FYZ
"""

import torch
import torch.nn as nn
from firstCNN import Net
from torch.autograd import Variable
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

net = Net()
optimizer = optim.SGD(net.parameters(), lr =0.1)

for i in range(1,30):
    input = torch.ones(3,1,32,32)+2
    output = net(input)
    target = Variable(torch.ones(3,10))
    criterion = nn.MSELoss()
    loss = criterion(output,target)
    print(loss)
    #print(loss.grad_fn)  # MSELoss
    #print(loss.grad_fn.next_functions[0][0])  # Linear
    #print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
