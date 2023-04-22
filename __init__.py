import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


transformer = nn.Transformer
lstm = nn.LSTM(input_size=21,hidden_size=24)
softmax = nn.Softmax(dim=1)
sig = nn.Sigmoid()
a = torch.tensor([0,1,2])
arr = torch.tensor([[1.,2,3],
                    [2,5,8]])
print(softmax(arr))
print(sig(a))

def f(x,y) ->float:
    return x+y
print(f(2.3,3))
a = False
print(type(a))
b = (1,2,3)
print(b,type(b))

def f(x):
    return np.exp(x)
# x = np.arange(-5,5,0.1)
x = np.linspace(-5,5,100)
y = f(x)

def g(x):
    return torch.exp(x)
x = torch.arange(-6,6,0.1)
y = g(x)

plt.plot(x,y)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

