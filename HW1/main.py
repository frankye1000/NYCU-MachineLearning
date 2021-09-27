import os
import numpy as np
from algo import lse, newton
from plot import print_fittingline_totalerror, plot

path = "D:\\NYCU\\NYCU-MachineLearning\\HW1"#input('path = ')
name = "testfile.txt"#input('name = ')
n = 3#int(input('n = '))
Lambda = 0#int(input('Lambda = '))

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


# lse
print("lse:")
parameters_lse, loss_lse = lse(A, b, n, Lambda)
print_fittingline_totalerror(n-1, parameters_lse, loss_lse)
print()

# Netwon's method
print("Newton's Method:")
parameters_newton, loss_newton = newton(A, b, n)
print_fittingline_totalerror(n-1, parameters_newton, loss_newton)

# plot
plot(x0, b, parameters_lse, parameters_newton)




# if __name__ == '__main__':
#     # iuput testfile data 
    
