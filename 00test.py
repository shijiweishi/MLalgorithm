import math
import numpy as np
import tensorflow
import torch
import torch.nn as nn
import cv2
from PIL import Image
import tensorflow as tf
import paddle
from paddle.nn import Conv2D, MaxPool2D, Linear
arr = torch.tensor([1,2,3,4,5,6,7,8])
arr1 = torch.reshape(arr,[2,4])
print(arr1)
arr2 = arr.reshape(4,2)
print(arr2)
print(11.6612/1.8382,-5.1837/1.8382)
print(torch.cuda.device_count())
# Python 3.X对于浮点数默认的是提供17位数字的精度
a = 3.145323222334478432
print(a)
print(type(a))
print(math.pi)
print(10/3)

b = 0.0000000000000012345678
c = 12345678912345678912345e3
print(b)
print(c)
d = (1,2,3)
print(d)
print(type(d))

# 元组和列表的区别
# 当希望内容不被轻易改写的时候，需要使用元组。如果数据需要频繁修改，那么需要使用列表。
# 列表提供了比元组更丰富的内置方法，这相当大的提高了编程的灵活性，元组固然安全，但元组一定创建就无法修改。
# 元组只有在特殊的情况才用到，平时还是列表用的多。

t1 = (1,2,3)
t2 = (2,3,4)
l1 = [1,1,1]
l2 = [2,2,2]
print(t1+t2)
print(l1+l2)
print(type(t2+t1))
print(type(l2+l1))

n1 = np.array([1,2,3])
n2 = np.array([4,5,6])
print(n1+n2)


# img = Image.open(r'D:\data\horse\101.jpg')
# img.show()
# img = np.array(img)#图片转矩阵
# print(img.shape)


# image = cv2.imread(r'D:\data\horse\100.jpg')#image :是返回提取到的图片的值
# cv2.imshow('image',image)     #展示BGR形式的图片
# cv2.waitKey()               # 默认为0，无限等待
# print(image.shape)


def f(x,y) -> int:
    return x+y
print(f(2,3.1))

# arr = tf.constant([[1,2,3]])
# print(arr)


a = paddle.to_tensor([
    [1,2,3],
    [2,3,4]
])
print(a)

# import flask
# paddle.utils.run_check()

arr = tf.constant([1,2,3])
print(arr)

a =np.array([1,5,9])
arr3 = paddle.to_tensor(a)
print(arr3)

relu = nn.ReLU()


from bushu.A import B
a = B()
a.f()
import bushu
from bushu.A import B