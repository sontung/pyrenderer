import numpy as np

from mathematics.fast_op import fast_dot3, cross_product2, fast_subtract


def func(u, v, g):
    fast_subtract(u, v, g)
    fast_subtract(u, v, g)
    fast_subtract(u, v, g)
    fast_subtract(u, v, g)
    fast_subtract(u, v, g)
    fast_subtract(u, v, g)
    fast_subtract(u, v, g)
    fast_subtract(u, v, g)
    fast_subtract(u, v, g)
    fast_subtract(u, v, g)

@profile
def main():
    u = np.ones((5,))
    v = np.ones((5,))
    g = np.ones((5,))
    fast_subtract(np.ones((1,)), np.ones((1,)), np.ones((1,)))
    func(u, v, g)
    res = [fast_subtract(u, v, g) for _ in range(10)]
    for _ in range(10):
        fast_subtract(u, v, g)

main()