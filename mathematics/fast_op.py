import numpy as np
from .constants import EPS
from numba import njit, prange, guvectorize, float64


@njit("(f8[:], f8[:], f8[:], u1[:], u1[:], f8, f8)")
def compute_t(a_e2r, f_vec, t_vec, first_compare, second_compare, bound, eps):
    for i in range(a_e2r.shape[0]//2):
        if -eps < a_e2r[i*2] < eps:
            first_compare[i] = 1
        else:
            f = 1/a_e2r[i*2]
            t = f*a_e2r[i*2+1]
            if t > bound or t < eps:
                second_compare[i] = 1
            else:
                t_vec[i] = t
                f_vec[i] = f


@njit("f8(f8[:], f8[:])")
def fast_dot(x, ty):
    res = x[0]*ty[0]+x[1]*ty[1]+x[2]*ty[2]
    return res


@njit("f8[:](f8[:], f8[:])")
def fast_dot3(x, ty):
    res = np.zeros((x.shape[0]//3), np.float64)
    for i in range(0, x.shape[0], 3):
        res[i//3] = x[i]*ty[i]+x[i+1]*ty[i+1]+x[i+2]*ty[i+2]
    return res


@njit("f8[:](f8[:], f8[:])")
def fast_subtract(x, ty):
    return x-ty


@njit("(f8[:], f8[:], f8[:])")
def fast_subtract2(x, ty, s):
    for i in range(3):
        s[i] = x[i]-ty[i]


@njit("(f8[:], f8[:], f8[:])")
def cross_product2(x, y, res):
    for i in range(res.shape[0]//3):
        res[i*3] = x[i*3+1]*y[i*3+2] - x[i*3+2]*y[i*3+1]
        res[i*3+1] = x[i*3+2]*y[i*3] - x[i*3]*y[i*3+2]
        res[i*3+2] = x[i*3]*y[i*3+1] - x[i*3+1]*y[i*3]


@guvectorize([(float64[:], float64[:], float64[:])], '(n),(n)->(n)')
def cross_product(x, y, res):
    for i in range(res.shape[0]//3):
        res[i*3] = x[i*3+1]*y[i*3+2] - x[i*3+2]*y[i*3+1]
        res[i*3+1] = x[i*3+2]*y[i*3] - x[i*3]*y[i*3+2]
        res[i*3+2] = x[i*3]*y[i*3+1] - x[i*3+1]*y[i*3]

