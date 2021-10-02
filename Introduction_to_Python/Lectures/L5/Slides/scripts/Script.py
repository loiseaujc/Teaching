import numpy as np
import numpy.random as npr
import numpy.linalg as npl

def distance(A, B):
    m, n = A.shape

    d = 0

    for i in range(m):
        for j in range(n):
            d += (A[i, j] - B[i, j])**2

    return np.sqrt(d)

def distance2(A, B):
    d = np.sum( (A-B)**2 )
    return np.sqrt(d)
