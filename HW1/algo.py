import numpy as np
from matrix_tool import inverse


def get_loss_value(A, x, b):
    return np.sum(np.square(A@x-b))


def lse(A, b, n, Lambda):
    x=inverse(A.T@A+Lambda*np.identity(n))@A.T@b

    loss_value=get_loss_value(A,x,b)
    return x,loss_value


def newton(A, b, n):
    # 初始 x
    x0 = np.random.rand(n, 1)
    eps = 100
    while eps>1e-6:
        x1 = x0 - inverse(2*A.T@A)@(2*A.T@A@x0-2*A.T@b)
        eps = abs(np.sum(np.square(x1-x0))/n)
        x0=x1

    loss_value = get_loss_value(A, x0, b)
    return x0, loss_value