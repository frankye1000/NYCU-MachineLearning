def factorial(x):
    if x == 0:
        return 1
    r = 1
    for i in range(1, x+1):
        r = r * i
    return r 


def combination(N, m):
    return int(factorial(N)/(factorial(m)*factorial(N-m)))

