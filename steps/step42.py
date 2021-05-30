import numpy as np
if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dezero import Variable
from dezero.utils import plot_dot_graph
import dezero.functions as F

# toy dataset
np.random.seed(0)
x = np.random.rand(100, 1)
y = 5 + 2 * x + np.random.rand(100, 1)
x, y = Variable(x), Variable(y)

W = Variable(np.zeros((1, 1)))
b = Variable(np.zeros(1))


def predict(x):
    y = F.matmul(x, W) + b
    return y


lr = 0.1
iters = 100

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    W.cleargrad()
    b.cleargrad()
    loss.backward()

    '''
    注意：パラメータの更新は単にデータを更新するだけなので，計算グラフを作る必要はなく，インスタンス変数のdataに対して計算を行う
    '''
    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data
    print(W, b, loss)
