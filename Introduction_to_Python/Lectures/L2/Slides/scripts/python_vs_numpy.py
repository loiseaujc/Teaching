import numpy as np
import timeit

def matvec(A, x):
    nrows, ncols = len(A), len(A[0])
    b = [sum(A[i][j] * x[j] for j in range(ncols)) for i in range(nrows)]
    return b

def matadd(A, B):
    nrows, ncols = len(A), len(A[0])
    C = [[A[i][j] + B[i][j] for j in range(ncols)] for i in range(nrows)]
    return C

def mattranpose(A):
    nrows, ncols = len(A), len(A[0])
    B = [[A[i][j] for i in range(nrows)] for j in range(ncols)]
    return B

if __name__ == "__main__":

    n = 10

    A = np.random.randn(n, n).tolist()
    B = np.random.randn(n, n).tolist()
    x = np.random.randn(n).tolist()

    Ap = np.random.randn(n, n)
    Bp = np.random.randn(n, n)
    xp = np.random.randn(n)

    ntrials = 100

    chrono = timeit.timeit(
    "matadd(A, B)",
    setup="from __main__ import matadd, A, B",
    number=ntrials
    ) / ntrials

    print("Pure Python matrix addition : {} seconds".format(chrono))

    chrono = timeit.timeit(
        "Ap + Bp",
        setup="from __main__ import Ap, Bp",
        number=ntrials
    ) / ntrials

    print("NumPy matrix addition : {} seconds".format(chrono))





    chrono = timeit.timeit(
    "mattranpose(A)",
    setup="from __main__ import mattranpose, A",
    number=ntrials
    ) / ntrials

    print("Pure Python matrix transpose : {} seconds".format(chrono))

    chrono = timeit.timeit(
        "Ap.T",
        setup="from __main__ import Ap",
        number=ntrials
    ) / ntrials

    print("NumPy matrix transpose : {} seconds".format(chrono))






    chrono = timeit.timeit(
    "matvec(A, x)",
    setup="from __main__ import matvec, A, x",
    number=ntrials
    ) / ntrials

    print("Pure Python matrix-vector : {} seconds".format(chrono))

    chrono = timeit.timeit(
        "Ap @ xp",
        setup="from __main__ import Ap, xp",
        number=ntrials
    ) / ntrials

    print("NumPy matrix-vector : {} seconds".format(chrono))
