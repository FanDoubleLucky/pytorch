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
import torchvision
import torchvision.transforms as transforms
net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()
'''
for epoch in range(1,100):
    input = torch.ones(1,3,32,32)+3
    out = net(input)
    
    target = Variable(torch.ones(1))
    loss = criterion(out,target.long())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss)
'''
if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)
    
    iter(trainloader)
    for epoch in range(2):  # 循环遍历数据集多次
    
        running_loss = 0.0
        
        for i,data in enumerate(trainloader,0):
            # 得到输入数据
            
            inputs, labels = data
    
            #pack = torch.Tensor(1,3,32,32)
            #pack[0] = inputs
            
            # 包装数据
            inputs, labels = Variable(inputs), Variable(labels)
            
            # 梯度清零
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels.long())
            print(i,loss)
            loss.backward()
            optimizer.step()
    
            # 打印信息
            running_loss += loss.data[0]
            if i % 2000 == 1999:    # 每2000个小批量打印一次
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    
    print('Finished Training')
