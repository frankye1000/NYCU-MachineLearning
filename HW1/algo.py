import numpy as np
from matrixTool import inverse

def matrixMul(A, B):
    return A.dot(B)


def lossValue(A, x, b):
    return np.sum(np.square( matrixMul(A, x) - b))


def lse(A, b, n, Lambda):
    x = matrixMul(matrixMul(inverse(matrixMul(A.T, A) + Lambda * np.identity(n)), A.T), b)    # 公式: x=(At*A+lambda*I)*At*b
    loss_value = lossValue(A, x, b)
    return x, loss_value


def newton(A, b, n):
    # 初始 x
    x0 = np.random.rand(n, 1)
    eps = 100
    while eps>1e-6: 
        x1 = x0 - matrixMul(inverse(2 * matrixMul(A.T, A)) , (2 * matrixMul(matrixMul(A.T, A), x0) - 2 * matrixMul(A.T, b)))    # 公式: x1 = x0-(2*A^t*A)^-1)*(2*A^t*A*x0-2*A^t*b) 
        eps = abs(np.sum(np.square(x1-x0))/n)    # 小於0.000001就停止
        x0 = x1

    loss_value = lossValue(A, x0, b)
    return x0, loss_value

