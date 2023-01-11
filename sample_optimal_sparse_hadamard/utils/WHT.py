import numpy as np
import sys
sys.path.append("..")
from sample_optimal_sparse_hadamard.fastwhtmain.python.hadamard import fastwht

def WHT(x):
    # print(x)
    n = len(x.shape)
    assert (x.shape == tuple([2] * n)), "x does not have the correct shape"
    N = 2 ** n
    # fastwht only works with numpy floats
    x = x.astype(dtype=np.float64)
    x = x.reshape((-1))
    # print("x=", x, "N=", N, "x shape=", x.shape)
    out = fastwht(x, N, 'hadamard')
    out = out.reshape(tuple([2] * n))
    return out


if __name__ == "__main__":
    np.random.seed(10)
    x = np.random.randint(0, high=5, size=(2, 2, 2))
    y = WHT(x)
    print(x)
    print(x[(0, 0, 0)])
    print(x[(0, 0, 1)])
    print(x[(0, 1, 0)])
    print(x[(0, 1, 1)])
    print(x[(1, 0, 0)])
    print(x[(1, 0, 1)])
    print(x[(1, 1, 0)])
    print(x[(1, 1, 1)])
    y = 8 * y
    print(y)
    print(y[(0, 0, 0)])
    print(y[(0, 0, 1)])
    print(y[(0, 1, 0)])
    print(y[(0, 1, 1)])
    print(y[(1, 0, 0)])
    print(y[(1, 0, 1)])
    print(y[(1, 1, 0)])
    print(y[(1, 1, 1)])
