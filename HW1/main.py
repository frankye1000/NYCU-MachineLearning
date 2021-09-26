import os
import numpy as np

path = "D:\\NYCU-MachineLearning\\HW1"#input('path= ')
name = "testfile.txt"#input('name= ')
poly_basis = 4#int(input('n= '))
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
    for i in range(poly_basis):
        r.append(v ** i)
    A.append(r)

A = np.array(A)
b = np.array(b).reshape((-1,1))
print(A)
print(b)


# if __name__ == '__main__':
#     # iuput testfile data 
    
