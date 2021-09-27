import numpy as np

def eliminate(r1, r2, col, target=0):
    fac = (r2[col]-target) / r1[col]
    for i in range(len(r2)):
        r2[i] -= fac * r1[i]


def gauss(a): 
    # 1. 把矩陣變上三角矩陣
    for i in range(len(a)):
        if a[i][i] == 0:        # 對角線是0，和下面一列交換
            for j in range(i+1, len(a)):
                if a[i][j] != 0:
                    a[i], a[j] = a[j], a[i]
                    break
            else:
                raise ValueError("Matrix is not invertible")
        
        for j in range(i+1, len(a)):
            eliminate(a[i], a[j], i)
    
    # 2. 把上三角形矩陣變對角矩陣(由下往上丟列)
    for i in range(len(a)-1, -1, -1):
        for j in range(i-1, -1, -1):
            eliminate(a[i], a[j], i)
    
    # 3. 把對角矩陣變單位矩陣
    for i in range(len(a)):
        eliminate(a[i], a[i], i, target=1)
    return a


def inverse(a):
    tmp = [[] for _ in a]
    for i, row in enumerate(a):
        assert len(row) == len(a)                             # 行、列同維
        row = list(row)                                       # array轉list
        tmp[i].extend(row + [0]*i + [1] + [0]*(len(a)-i-1))   # 補identity matrix於右邊
    gauss(tmp)
    ret = []
    for i in range(len(tmp)):
        ret.append(tmp[i][len(tmp[i])//2:])
    return np.array(ret)