#三层神经网络


import math
#sigmoid函数，参数deriv为True时，表示求导数，否则将一个实数转化为0到1间的一个数，也可以叫概率。
import numpy as np
def sigmoid(x,deriv=False):#deriv导数，默认为FALSE
    if deriv == True:
        return x*(1-x)
    return 1/(1+np.exp(-x))
x=np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1],
            [0,0,1]])#5*3
y=np.array([[0],
            [1],
            [0],
            [1],
            [0]])#5*1
np.random.seed(1)#指定随机种子
w0=2*np.random.random((3,4))-1#权重参数，每个元素在-1到1之间[-1,1)
w1=2*np.random.random((4,1))-1#权重参数
for j in range(100000):
    L0=x
    L1=sigmoid(np.dot(L0,w0))
    L2=sigmoid(np.dot(L1,w1))
    # print(L2)
    L2_error=L2-y
    if j%10000 == 0:
        print('Error '+str(np.mean(np.abs(L2_error))))#np.mean()求取均值,np.abs()函数用于返回数字的绝对值
    #反向传播(反向传播就是为了实现最优化，省去了重复的求导步骤)
    L2_delta = L2_error * sigmoid(L2,deriv=True)#矩阵A*B是表示对应元素相乘，不是矩阵乘积
    # print(sigmoid(L2,deriv=True)) #L2的5个元素的导数
    # print(L2_delta)
    L1_error = L2_delta.dot(w1.T)#等价于np.dot(L2_delta,w1.T)#5*4矩阵
    # print(L1_error)
    L1_delta = L1_error * sigmoid(L1,deriv=True)
    # 更新w 参数
    w1 -= L1.T.dot(L2_delta)
    w0 -= L0.T.dot(L1_delta)
print('L1： {}'.format(L1))
print('L2： {}'.format(L2))
