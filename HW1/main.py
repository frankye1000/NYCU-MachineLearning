import os
import numpy as np
from newton import newton
from plot import print_fittingline_totalerror

path = "D:\\NYCU\\NYCU-MachineLearning\\HW1"#input('path= ')
name = "testfile.txt"#input('name= ')
n = 4#int(input('n= '))
Lambda = 2#int(input('Lambda= '))

# 讀資料
x0=[]
b=[]
filepath = os.path.join(path, name)
with open(filepath) as f:
    for line in f.readlines():
        s = line.split(',')
        x0.append(float(s[0]))
        b.append(float(s[1]))
# 做矩陣A
A = []
for v in x0:
    r = []
    for i in range(n):
        r.append(v ** i)
    A.append(r)

A = np.array(A)
b = np.array(b).reshape((-1,1))
print(A)
print(b)


# Netwon's method
parameters_newton, loss_newton = newton(A,b)
print('Newton\'s Method:')
print_fittingline_totalerror(n-1, parameters_newton, loss_newton)
# plot(x1.reshape(-1), b.reshape(-1), parameters_rlse.reshape(-1),parameters_newton.reshape(-1))




# if __name__ == '__main__':
#     # iuput testfile data 
    
