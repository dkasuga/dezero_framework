import numpy as np
if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dezero import Variable
from dezero.utils import plot_dot_graph
import dezero.functions as F

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = x.T
y.backward(retain_grad=True)
print(x.grad)
