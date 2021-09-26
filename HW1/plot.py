import numpy as np
import matplotlib.pyplot as plt


def print_fittingline_totalerror(n, parameters, loss):
    print("Fitting line: ", end = '')
    for i, parameter in enumerate(parameters):
        if n-i == 0:
            print(parameter[0])
        else:
            print(str(parameter[0]) + "X^{}".format(n-i) + ' + ', end='')

    print('Total error: ', loss)


def plot(x1,b,parameters_rlse,parameters_newton):
    #rlse
    plt.subplot(2,1,1)
    plt.title('rlse')
    plt.plot(x1,b,'ro')
    x1_min=min(x1)
    x1_max=max(x1)
    x=np.linspace(x1_min-1,x1_max+1,500)
    y=np.zeros(x.shape)
    for i in range(len(parameters_rlse)):
        y+=parameters_rlse[i]*np.power(x,i)
    plt.plot(x,y,'-k')
    #newton
    plt.subplot(2,1,2)
    plt.title('newton\'s method')
    plt.plot(x1, b, 'ro')
    y = np.zeros(x.shape)
    for i in range(len(parameters_newton)):
        y += parameters_newton[i] * np.power(x, i)
    plt.plot(x, y, '-k')
    plt.show()