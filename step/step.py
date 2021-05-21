import numpy as np
import weakref
import contextlib

if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dezero import Variable

x = Variable(np.array(1.0))
y = (x + 3) ** 2
y.backward()

print(y)
print(x.grad)


# class Square(Function):
#     def forward(self, x):
#         y = x ** 2
#         return y

#     def backward(self, gy):
#         x = self.inputs[0].data
#         gx = 2 * x * gy
#         return gx


# class Exp(Function):
#     def forward(self, x):
#         return np.exp(x)

#     def backward(self, gy):
#         x = self.inputs[0].data
#         gx = np.exp(x) * gy
#         return gx


# def square(x):
#     f = Square()
#     return f(x)


# def exp(x):
#     f = Exp()
#     return f(x)
