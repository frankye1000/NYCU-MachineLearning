import numpy as np
import matplotlib.pyplot as plt


def print_fittingline_totalerror(n, parameters, loss):
    print("Fitting line: ", end = '')

    parameters = list(parameters)[::-1]
    for i, parameter in enumerate(parameters):
        if n-i == 0:
            print(parameter[0])
        else:
            print(str(parameter[0]) + "X^{}".format(n-i) + ' + ', end='')

    print('Total error: ', loss)


def plot(x0, b, parameters_lse, parameters_newton):
    # lse
    plt.subplot(2, 1, 1)
    plt.title('lse')
    plt.plot(x0, b, 'ro')
    x0_min = min(x0)
    x0_max = max(x0)   
    x = np.arange(x0_min-1, x0_max+1, 0.1)          # x軸點數(畫線)
    y = np.zeros(x.shape)
    for i in range(len(parameters_lse)):
        y += parameters_lse[i]*np.power(x,i)        # y軸數值(依照fittingline計算)
    plt.plot(x, y, '-k')

    #newton
    plt.subplot(2, 1, 2)
    plt.title("newton's method")
    plt.plot(x0, b, 'ro')
    y = np.zeros(x.shape)
    for i in range(len(parameters_newton)):
        y += parameters_newton[i] * np.power(x, i)
    plt.plot(x, y, '-k')
    plt.show()