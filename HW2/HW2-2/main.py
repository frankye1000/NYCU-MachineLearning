import os
from function import combination

path = "D:\\NYCU\\NYCU-MachineLearning\\HW2\\HW2-2"#input('path: ')
name = "testfile.txt"#input('name: ')
a = 0#int(input('a: '))
b = 0#int(input('b: '))

filepath = os.path.join(path, name)

with open(filepath, "r") as f:
    cases = f.readlines()

# 數有多少個1
def count_m(case):
    m = 0
    for i in case:
        if i == "1":
            m += 1
    return m


for i, case in enumerate(cases):
    case = case.strip()   # 去除換行符號
    N = len(case)
    m = count_m(case)
    p = m / N
    likelihood = combination(N, m)*p**m*(1-p)**(N-m)
    print("case {}: {}".format(i + 1, case))
    print("Likelihood: {}".format(likelihood))
    print("Beta prior:     a = {} b = {}".format(a, b))
    a += m
    b += (N-m)
    print("Beta posterior: a = {} b = {}".format(a, b))
    print()




