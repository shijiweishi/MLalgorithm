# 隐层rulu加均方误差;输出层softmax加交叉熵;反向传播
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import matplotlib.pyplot as plt
from torch import nn
def sigmoid(x,deriv=False):#deriv导数，默认为FALSE
    if deriv == True:
        return x*(1-x)
    return 1/(1+np.exp(-x))
def relu(x):
    return np.maximum(0,x)

def relu_grad(x):
    y = []
    for i in x:
        if i >= 0:
            y.append(1)
        else:
            y.append(0)
    return y

# 二维数组的relu梯度函数
def relu_grad2(x):
    y = x.tolist()
    for i,data1 in enumerate (y):
        for j,data2 in enumerate (data1):
            if data2 >= 0:
                y[i][j] = 1
            else:
                y[i][j] = 0
    y = np.array(y)
    return y
'''
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))
'''
softmax = nn.Softmax(dim=1)
x = np.array([[0.,0,1,2,3,2,6,7],
            [0,1,1,2,1,8,2,6],
            [1,0,1,1,3,5,8,12],
            [1,1,1,4,2,4,9,13],
            [0,0,1,4,1,4,9,12]])#5*8;输入为5个样本，8个特征
y = np.array([[1],
             [2],
             [3],
             [4],
             [2]])#5*1;对应标签值;假设只有四个标签1,2,3,4。最后输出应该是1个样本对应4个概率值
np.random.seed(1)#指定随机种子
# 设置两个隐层
w1 = 2*np.random.random((8,4))-1#权重参数，每个元素在-1到1之间[-1,1)
w2 = 2*np.random.random((4,4))-1#权重参数
print('w1 :',w1)
print('w2 :',w2)

def f(x):
    if x == True:
        return 1
    else:
        return 0
# 对y独热编码;变成Y
len_y = y.shape[0]
Y = []
for i in range(len_y):
    # print(y[i][0])
    one_hot_y = [f(y[i][0]==1),f(y[i][0]==2),f(y[i][0]==3),f(y[i][0]==4)]
    Y.append(one_hot_y)
Y = torch.tensor(Y)
print('Y',Y)



y = torch.tensor(y)
y = torch.reshape(y,[5])
print(y.dtype)
y = y.type(torch.LongTensor)
y = y-1
print(y)
print(y.dtype)
print('+++++++++++++++++')
losses = []
for i in range(10000):
    #print(np.dot(x,w1))
    L0 = torch.tensor(x)
    w1 = torch.tensor(w1)
    w2 = torch.tensor(w2)
    ReLU = nn.ReLU()
    L1 = ReLU(torch.matmul(L0,w1))  #5*4
    #print(L1)
    L2 = torch.matmul(L1,w2)    #5*4
    #print(L2)

    crossentropyloss=nn.CrossEntropyLoss(reduction='none')  #其实CrossEntropyLoss相当于softmax + log + nllloss。
    crossentropyloss_output=crossentropyloss(L2,y)
    #print(crossentropyloss_output)
    loss = torch.mean(crossentropyloss_output)  #交叉熵损失值
    losses.append(loss)
    if i % 200 == 0:
        print('loss',loss)
    if i == 9999:
        print(L2)
        print(softmax(L2))
    # 交叉熵损失函数梯度
    softmax = nn.Softmax(dim=1)
    output1 = softmax(L2)
    #print(output1)
    #print(output1-Y)
    grad_w2 = (output1-Y)/5     #grad_w2相当于 L2_delta;5*4
    #print(grad_w2)

    L1_error = torch.matmul(grad_w2,w2.T)
    L1_delta = L1_error * relu_grad2(L1)

    w2 = w2 - torch.mm(L1.T,grad_w2)*0.001
    w1 = w1 - torch.mm(L0.T,L1_delta)*0.001
x = np.arange(len(losses))
y = losses
plt.plot(x,y)
plt.show()

print('w1',w1)
print('w2',w2)
torch.save(w1,'w1')
torch.load('w1')