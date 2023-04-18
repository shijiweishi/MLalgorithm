import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd


# sigmoid = nn.Sigmoid()
def sigmoid(x):
    return 1/(1+np.exp(-x))

# x = np.arange(-5,5,0.1)
# y = sigmoid(x)
# plt.plot(x,y)
# plt.grid(True)
# plt.show()

# arr = torch.tensor([1,0])
# print(sigmoid(arr))
file = r'D:\data\testSet.txt'
file = r'D:\data\LR.txt'


data = pd.read_csv(file,sep='\s+')
data = data.to_numpy()

# print(data)
# print(data.shape)

# 绘制散点图
# color=[]
# for i in range(len(data)):
#     if data[i][2] == 1:
#         color.append('red')
#     else:
#         color.append('blue')
# plt.scatter(data[:,0],data[:,1],color=color)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

x = data[:,0:2]
print(x)
y = data[:,-1]
print(y)
print(y.shape)


#   初始化参数w,b
w = np.random.randn(2,1)
b = np.random.randn()
print(w)
print(b)

'''
for j in range(80000):
    x = data[:, 0:2]
    y = data[:, -1]
    # 计算wx+b,然后计算sigmoid(wx+b)
    z = np.dot(x,w)+b
    z = sigmoid(z)
    # print(z)

    #   交叉熵计算损失
    L = 0
    for i in range(len(y)):
        loss = -y[i]*np.log(z[i])-(1-y[i])*np.log(1-z[i])
        L = L + loss
    L = L/len(y)

    if j % 500 == 0:
        print('j:{}, 交叉熵计算的损失:{}'.format(j,L))

    #   梯度下降法更新梯度,学习率0.01
    #   计算梯度
    grad_w = 0
    grad_b = 0
    for i in range(len(y)):
        g_w = (z[i]-y[i])*x[i]
        grad_w = grad_w + g_w
        g_b = (z[i]-y[i])
        grad_b = grad_b + g_b
    grad_w = grad_w / len(y)
    grad_b = grad_b / len(y)
    # print('grad_w',grad_w)
    # print(grad_w.shape)
    grad_w = np.reshape(grad_w,[2,1])
    # print(grad_w)
    # # 更新梯度w,b
    w = w - grad_w * 0.015
    b = b - grad_b * 0.015
    if j == 79999:
        np.save('w',w)
        np.save('b',b)
'''
# w = np.load('w.npy')
# b = np.load('b.npy')
# print(w,b)

# 散点图和逻辑回归的直线:
data = pd.read_csv(file,sep='\s+')
data = data.to_numpy()

color=[]
for i in range(len(data)):
    if data[i][2] == 1:
        color.append('red')
    else:
        color.append('blue')
plt.scatter(data[:,0],data[:,1],color=color)
plt.xlabel('x')
plt.ylabel('y')
x = np.arange(-4,4,0.1)
y = -2.8202 * x + 6.344
plt.plot(x,y)
plt.show()