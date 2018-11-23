# -*- coding: utf-8 -*-

import torch

x = torch.ones(5, 5).type(torch.IntTensor)
print torch.mm(x, x)
print x.mm(x)

y = torch.randn(5, 5).type(torch.FloatTensor)
print type(y)

dtype = torch.FloatTensor
N, D_in, H, D_out = 64, 1000, 100, 10

# 创建随机输入和输出数据
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# 随机初始化权重
w1 = torch.randn(D_in, H).type(dtype)
w2 = torch.randn(H, D_out).type(dtype)

learning_rate = 1e-6
for t in range(50):
    # 正向传递：计算预测y
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    # 计算并打印loss
    loss = (y_pred - y).pow(2).sum()
    # print(t, loss)

    # 反向传播计算关于损失的w1和w2的梯度
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # 使用梯度下降更新权重
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2