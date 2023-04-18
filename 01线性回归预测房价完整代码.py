#导入数据
import numpy as np
import matplotlib.pyplot as plt
#梯度下降法
'''
#数据处理封装
def load_data():
    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT',
                     'MEDV']
    feature_numbers = len(feature_names)
    # 数据读取,数据变形
    data_file = 'D:\数据集\housing.data'
    data = np.fromfile(data_file, sep=' ')
    data = data.reshape([data.shape[0] // feature_numbers, feature_numbers])  # 两个整数相除之后自动变成float类型，要向下取整

    # 数据归一化
    maxs = data.max(axis=0)  # 取每一列的最大值
    mins = data.min(axis=0)  # 取每一列的最小值
    avgs = data.sum(axis=0) / data.shape[0]
    for i in range(feature_numbers):
        data[:, i] = (data[:, i] - avgs[i]) / (maxs[i] - mins[i])

    # 数据集划分
    ratio = 0.8
    off = int(data.shape[0] * ratio)
    train_data = data[:off]
    test_data = data[off:]
    return train_data,test_data

#模型设计，前向传播
class Net(object):
    def __init__(self,num_weight):
        # 为了保持程序每次运行结果的一致性，此处设置固定的随机数种子
        np.random.seed(0)
        self.w = np.random.randn(num_weight,1)
        self.b = 0
    #前向传播
    def forward(self,x):
        z = np.dot(x,self.w) + self.b
        return z
    #计算损失
    def loss(self,z,y):
        error = z - y
        cost = error * error
        cost = np.mean(cost)
        return cost
    #计算梯度
    def gradient(self,x,y):
        z = self.forward(x)
        gradient_w = (z - y) * x
        gradient_w = np.mean(gradient_w,axis=0)
        gradient_w = gradient_w[:,np.newaxis]
        gradient_b = z - y
        gradient_b = np.mean(gradient_b,axis=0)
        return gradient_w,gradient_b
    #梯度更新
    def update(self,gradient_w,gradient_b,eta=0.01):
        self.w = self.w - eta * gradient_w
        self.b = self.b - eta * gradient_b
    def train(self,x,y,iterations=100,eta=0.01):
        losses = []
        for i in range(iterations):
            z = self.forward(x)
            L = self.loss(z, y)
            losses.append(L)
            gradient_w, gradient_b = self.gradient(x, y)
            self.update(gradient_w, gradient_b,eta)
            if i % 20 == 0:
                print('iter:{},loss:{}'.format(i,L))
        return losses

train_data,test_data = load_data()
x = train_data[:,:-1]
y = train_data[:,-1:]
print(train_data.shape)
print(test_data.shape)

net = Net(13)
iter = 1000
losses = net.train(x,y,iter,eta=0.01)
plot_x = np.arange(iter)
plot_y = losses
plt.plot(plot_x,plot_y)
plt.show()
print(net.w)
print(net.b)
'''


#随机梯度下降
def load_data():
    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT',
                     'MEDV']
    feature_numbers = len(feature_names)
    # 数据读取,数据变形
    data_file = 'D:\data\housing.data'
    data = np.fromfile(data_file, sep=' ')
    data = data.reshape([data.shape[0] // feature_numbers, feature_numbers])  # 两个整数相除之后自动变成float类型，要向下取整

    # 数据归一化
    maxs = data.max(axis=0)  # 取每一列的最大值
    mins = data.min(axis=0)  # 取每一列的最小值
    avgs = data.sum(axis=0) / data.shape[0]
    for i in range(feature_numbers):
        data[:, i] = (data[:, i] - avgs[i]) / (maxs[i] - mins[i])

    # 数据集划分
    ratio = 0.8
    off = int(data.shape[0] * ratio)
    train_data = data[:off]
    test_data = data[off:]
    return train_data,test_data

#模型设计，前向传播
class Net(object):
    def __init__(self,num_weight):
        # 为了保持程序每次运行结果的一致性，此处设置固定的随机数种子
        np.random.seed(0)
        self.w = np.random.randn(num_weight,1)
        self.b = 0
    #前向传播
    def forward(self,x):
        z = np.dot(x,self.w) + self.b
        return z
    #计算损失
    def loss(self,z,y):
        error = z - y
        cost = error * error
        cost = np.mean(cost)
        return cost
    #计算梯度
    def gradient(self,x,y):
        z = self.forward(x)
        gradient_w = (z - y) * x
        gradient_w = np.mean(gradient_w,axis=0)
        gradient_w = gradient_w[:,np.newaxis]
        gradient_b = z - y
        gradient_b = np.mean(gradient_b,axis=0)

        # 下面是矩阵的形式表示
        # z = self.forward(x)
        # N = x.shape[0]
        # A = np.ones([N, 1])
        # gradient_w = x.T.dot(z - y) / N
        # gradient_b = A.T.dot(z - y) / N
        return gradient_w,gradient_b
    #梯度更新
    def update(self,gradient_w,gradient_b,eta=0.01):
        self.w = self.w - eta * gradient_w
        self.b = self.b - eta * gradient_b
    def train(self,train_data,epoch=10,batch_size=10,eta=0.01):
        n = len(train_data)
        losses = []
        for i in range(epoch):
            np.random.shuffle(train_data)
            mini_batchs = [train_data[k:k + batch_size] for k in range(0, n, batch_size)]
            for iert_id,mini_batch in enumerate(mini_batchs):
                x = mini_batch[:,:-1]
                y = mini_batch[:,-1:]
                z = self.forward(x)
                loss = self.loss(z, y)
                gradient_w, gradient_b = self.gradient(x,y)
                self.update(gradient_w, gradient_b,eta=0.01)
                losses.append(loss)
                print('epoch/{},iter {},loss {}'.format(i,iert_id,loss))
        return losses


train_data,test_data = load_data()
print(train_data.shape)
# x = train_data[:,:-1]
# y = train_data[:,-1:]
# print(train_data.shape)
# print(test_data.shape)
# batch_size = 10
# mini_batchs = [train_data[k:k+10] for k in range(0,n,batch_size)]
# print('mini_batchs_first_shape',mini_batchs[0].shape)
# print('mini_batchs_last_shape',mini_batchs[-1].shape)

net = Net(13)
batch_size = 100
losses = net.train(train_data,epoch=200, batch_size=100, eta=0.01)
plot_x = np.arange(len(losses))
plt.plot(plot_x,losses)
plt.show()
w = net.w
b = net.b
print(w)
print(b)

# np.savez('p1',w,b)
# data = np.load('p1.npz')
# print(data)
# for i in data:
#     print(data[i])