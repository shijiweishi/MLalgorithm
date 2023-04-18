import torch
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

'''
# 逻辑回归在cpu上训练
# 1:数据处理
# sigmoid = nn.Sigmoid()
# arr = torch.tensor([1,0,2])
# print(sigmoid(arr))

file = r'D:\data\LR.txt'
data = pd.read_csv(file,sep='\s+')

# 要先加上一步转numpy然后才能转为torch张量
data = data.to_numpy()
# print(data)

data = torch.tensor(data)
print(data)

x = data[:,0:2]
x = x.to(torch.float32)
y = data[:,2]
# print('y:{},y.shape{}'.format(y,torch.Size(y)))
# print(len(y))
# print(y.shape)
# y = torch.reshape(y,[len(y),1])
y = y.reshape([len(y),1])
print(x,y)
print(y.shape)

# 2:构建模型，前向传播
class LRmodel(nn.Module):
    def __init__(self):
        super(LRmodel, self).__init__()
        self.fc = nn.Linear(in_features=2,out_features=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,input):
        output1 = self.fc(input)
        output2 = self.sigmoid(output1)
        return output2

model = LRmodel()

# 3:定义损失函数和优化器
criterion1 = nn.CrossEntropyLoss()      # CrossEntropyLoss相当于softmax + log + nllloss
criterion2 = nn.BCELoss()   # 对于分类问题，推荐采用二分类交叉熵函数;CEWithLogitsLoss() = sigmoid() + BCELoss()
opt = torch.optim.SGD(model.parameters(),0.01)


# # 4:开始迭代训练
for i in range(80000):
    opt.zero_grad()
    output = model.forward(x)
    loss = criterion2(output,y.float())
    loss.backward()
    opt.step()
    if i % 100 == 0:
        print('i:{},loss:{}'.format(i,loss))


torch.save(model.state_dict(), 'model_state')
model_state = torch.load('model_state')
print(model_state)


# 散点图和逻辑回归的直线:
# data = pd.read_csv(file,sep='\s+')
# data = data.to_numpy()
# data = torch.tensor(data)
# color=[]
# for i in range(len(data)):
#     if data[i][2] == 1:
#         color.append('red')
#     else:
#         color.append('blue')
# plt.scatter(data[:,0],data[:,1],color=color)
# plt.xlabel('x')
# plt.ylabel('y')
# x = torch.arange(-4,4,0.1)
# y = -2.949 * x + 6.53223
# plt.plot(x,y)
# plt.show()
'''




# 逻辑回归在gpu训练
# 使用GPU训练时，数据、函数和模型都必须同时放在GPU上，否则会出错。
# 1:数据处理
# sigmoid = nn.Sigmoid()
# arr = torch.tensor([1,0,2])
# print(sigmoid(arr))

file = r'D:\data\LR.txt'
data = pd.read_csv(file,sep='\s+')

# 要先加上一步转numpy然后才能转为torch张量
data = data.to_numpy()
# print(data)

data = torch.tensor(data)
print(data)

x = data[:,0:2]
x = x.to(torch.float32)
y = data[:,2]


# print('y:{},y.shape{}'.format(y,torch.Size(y)))
# print(len(y))
# print(y.shape)
# y = torch.reshape(y,[len(y),1])
y = y.reshape([len(y),1])
print(x,y)
print(y.shape)

# 数据在gpu上训练
x = x.cuda()
y = y.cuda()

# 2:构建模型，前向传播
class LRmodel(nn.Module):
    def __init__(self):
        super(LRmodel, self).__init__()
        self.fc = nn.Linear(in_features=2,out_features=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,input):
        output1 = self.fc(input)
        output2 = self.sigmoid(output1)
        return output2

model = LRmodel()
# 将网络模型在gpu上训练
model = model.cuda()

# 3:定义损失函数和优化器
criterion1 = nn.CrossEntropyLoss()      # CrossEntropyLoss相当于softmax + log + nllloss
criterion2 = nn.BCELoss()   # 对于分类问题，推荐采用二分类交叉熵函数;CEWithLogitsLoss() = sigmoid() + BCELoss()
opt = torch.optim.SGD(model.parameters(),0.01)


# # 4:开始迭代训练
# for i in range(120000):
#     opt.zero_grad()
#     output = model.forward(x)
#     loss = criterion2(output,y.float())
#     # 损失函数在gpu上训练
#     if torch.cuda.is_available():
#         loss = loss.cuda()
#     loss.backward()
#     opt.step()
#     if i % 500 == 0:
#         print('i:{},loss:{}'.format(i,loss))


# torch.save(model.state_dict(), 'model_state')
model_state = torch.load('model_state')
print(model_state)
print(model_state['fc.weight'],model_state['fc.bias'])

# 加载模型参数推理预测
new_model = LRmodel()
new_model.load_state_dict(torch.load('model_state'))        # 加载模型参数
input = torch.tensor([3,1.2])
out = new_model.forward(input)
print(out)


# 散点图和逻辑回归的直线:
data = pd.read_csv(file,sep='\s+')
data = data.to_numpy()
data = torch.tensor(data)
color = []
for i in range(len(data)):
    if data[i][2] == 1:
        color.append('red')
    else:
        color.append('blue')
plt.scatter(data[:,0],data[:,1],color=color)
plt.xlabel('x')
plt.ylabel('y')
x = torch.arange(-4,4,0.1)
y = -2.82 * x + 6.3438
plt.plot(x,y)
plt.show()