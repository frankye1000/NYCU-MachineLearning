import sys
import numpy as np

# n是矩陣order，使用Guass Jordan Elimination
# 這邊要擴充成
# 1 2 3 | 1 0 0
# 4 5 6 | 0 1 0
# 7 8 9 | 0 0 1

A=[[0,1,0],[1,1,0],[1,0,1]]
n=3
def inverse(n,A):
    I = np.identity(n)
    A = np.hstack([A, I])
    
    for i in range(n):
        if A[i][i] == 0.0:
            sys.exit('Divide by zero detected!')
            
        for j in range(n):
            if i != j:
                ratio = A[j][i]/A[i][i]

                for k in range(2*n):
                    A[j][k] = A[j][k] - ratio * A[i][k]

    # Row operation to make principal diagonal element to 1
    for i in range(n):
        divisor = A[i][i]
        for j in range(2*n):
            A[i][j] = A[i][j]/divisor
    print(A)
    # 回傳後面已處理好的反矩陣
    return np.hsplit(A,[n])[1]  
inverse(n,A)