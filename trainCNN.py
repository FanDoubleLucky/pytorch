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


def printnorm(self, input, output):
    # input是将输入打包成的 tuple 的input
    # 输出是一个 Variable. output.data 是我们感兴趣的 Tensor
    print('Inside ' + self.__class__.__name__ + ' forward')
    print('')
    print('input: ', type(input))
    print('input[0]: ', type(input[0]))
    print('output: ', type(output))
    print('')
    print('input size:', input[0].size())
    print('output size:', output.data.size())
    print('output norm:', output.data.norm())
    
    
net = Net()
optimizer = optim.SGD(net.parameters(), lr =0.1)

for i in range(1,20):
    input = torch.ones(3,1,32,32)+2
    net.conv2.register_forward_hook(printnorm)
    output = net(input)
    target = Variable(torch.Tensor(3,10))
    criterion = nn.MSELoss()
    loss = criterion(output,target)
    print(loss)
    #print(loss.grad_fn)  # MSELoss
    #print(loss.grad_fn.next_functions[0][0])  # Linear
    #print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
