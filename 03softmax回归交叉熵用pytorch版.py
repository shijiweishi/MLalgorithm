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

# def softmax(x):
#     if x.ndim == 2:
#         x = x.T
#         x = x - np.max(x, axis=0)
#         y = np.exp(x) / np.sum(np.exp(x), axis=0)
#         return y.T
#     x = x - np.max(x)
#     return np.exp(x) / np.sum(np.exp(x))


x = torch.tensor([[0.,0,1,2,3,2,6,7],
            [0,1,1,2,1.,8,2,6],
            [1,0,1,1,3,5,8,12],
            [1,1,1,4,2,4,9,13],
            [0,0,1,4,1,4,9,12]])#5*8;输入为5个样本，8个特征
y = np.array([[1],
             [2],
             [3],
             [4],
             [2]])#5*1;对应标签值;假设只有四个标签1,2,3,4。最后输出应该是1个样本对应4个概率值

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

softmax = nn.Softmax(dim=1)
Relu = nn.ReLU()
Sigmoid = nn.Sigmoid()
# Step 2:============================定义一个Model的模型===================
class Model(torch.nn.Module):
    def __init__(self,name):
        super(Model, self).__init__()
        self.name = name
        self.forward1 = torch.nn.Linear(in_features=8, out_features=4)
        self.forward2 = torch.nn.Linear(in_features=4, out_features=4)
    def forward(self, input):
        output1 = self.forward1(input)
        output1 = Relu(output1)
        output2 = self.forward2(output1)
        return output2
model = Model(x)
# Step 3:============================定义损失函数和优化器===================
# 定义 loss 函数，这里用的是交叉熵损失函数(Cross Entropy)
criterion = torch.nn.CrossEntropyLoss(reduction='none')
# 我们优先使用Adam下降，lr是学习率: 0.1
optimizer = torch.optim.Adam(model.parameters(),0.05)


losses = []
for i in range(5000):
    optimizer.zero_grad()
    crossentropyloss=nn.CrossEntropyLoss(reduction='none')  #其实CrossEntropyLoss相当于softmax + log + nllloss。
    output = model.forward(x)
    crossentropyloss_output=crossentropyloss(output,y)
    #crossentropyloss_output = criterion(output,y)

    loss = torch.mean(crossentropyloss_output)  #交叉熵损失值
    print('交叉熵损失:{} loss:{}'.format(crossentropyloss_output,loss))
    if i % 100 == 0:
        print('+++++++++++++++++++++++++')
        print('loss',loss)

    loss.backward()
    optimizer.step()
    losses.append(loss.tolist())
    if i == 4999:
        print(softmax(output))

x = np.arange(len(losses))
y = losses
plt.plot(x,y)
plt.grid()
plt.show()


# 模型的保存与读取
#1、保存整个模型
# torch.save(model,'model')
# model_dict = torch.load('model')
# print(model_dict)

# 2、仅保存模型的state_dict()、读取:
torch.save(model.state_dict(), 'model_state')
state_dict = torch.load('model_state')
print(state_dict)
print('forward1.weight',state_dict['forward1.weight'])
print('forward2.weight',state_dict['forward2.weight'])

w = torch.tensor([1,2,3])
# 3、保存训练好的权重矩阵w
torch.save(w, 'w')
w = torch.load('w')
print(w)


'''
# 1,2,3,4,2
# 传入数据预测
# w = torch.load('model_state')
# print(w)
x = torch.tensor([[0.,0,1,2,3,2,6,7],
            [0,1,1,2,1.,8,2,6],
            [1,0,1,1,3,5,8,12],
            [11,12,1,4,22,4,9,1],
            [0,0,1,4,1,4,9,12]])#5*8;输入为5个样本，8个特征
# print(model.forward(x))

new_model = Model('z')                                     # 调用模型Model
new_model.load_state_dict(torch.load('model_state'))    # 加载模型参数
print(new_model.forward(x))
'''