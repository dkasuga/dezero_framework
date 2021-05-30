import numpy as np
if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dezero import Variable
from dezero.utils import plot_dot_graph
import dezero.functions as F

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = Variable(np.array([7, 8, 9]))
z = x + y
z.backward(retain_grad=True)
print(x.grad)
print(y.grad)
