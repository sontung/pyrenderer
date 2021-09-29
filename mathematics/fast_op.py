import numpy as np
from .constants import EPS
from numba import njit, prange, guvectorize, float64
import numba
numba.config.NUMBA_DEFAULT_NUM_THREADS = 3


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


@njit("(f8[:], i1[:])")
def not_zeros(x, res):
    for i in range(x.shape[0]):
        if x[i] < -EPS or x[i] > EPS:
            res[i] = 1


@njit("(f8[:], i1[:], f8, f8)")
def in_bounds(x, res, b1, b2):
    for i in range(x.shape[0]):
        if b2 > x[i] > b1:
            res[i] = 1


@njit("UniTuple(i8[:], 1)(i1[:])")
def nonzero(x):
    return np.nonzero(x)


@njit("(f8[:],)")
def inv(x):
    for i in range(x.shape[0]):
        x[i] = 1.0/x[i]


@njit("(f8[:], f8[:])")
def multiply(x, y):
    for i in range(x.shape[0]):
        x[i] = x[i]*y[i]


@njit("f8(f8[:], f8[:])")
def fast_dot(x, ty):
    res = x[0]*ty[0]+x[1]*ty[1]+x[2]*ty[2]
    return res


@njit("(f8[:], f8[:], f8[:])")
def fast_dot3(x, ty, res):
    for i in range(0, x.shape[0], 3):
        res[i//3] = x[i]*ty[i]+x[i+1]*ty[i+1]+x[i+2]*ty[i+2]


@guvectorize([(float64[:], float64[:], float64[:])], '(n),(n)->(n)')
def fast_dot3_vectorized(x, ty, res):
    shape = res.shape[0] // 3
    for i in prange(shape):
        idx = i*3
        res[i] = x[idx]*ty[idx]+x[idx+1]*ty[idx+1]+x[idx+2]*ty[idx+2]


@njit("(f8[:], f8[:], f8[:])")
def fast_subtract(x, ty, s):
    for i in range(x.shape[0]):
        s[i] = x[i]-ty[i]


@guvectorize([(float64[:], float64[:], float64[:])], '(n),(n)->(n)')
def fast_subtract_vectorized(x, y, res):
    for i in range(x.shape[0]):
        res[i] = x[i]-y[i]


@njit("(f8[:], f8[:], f8[:])")
def cross_product(x, y, res):
    shape = res.shape[0] // 3
    for i in range(shape):
        idx = i*3
        res[idx] = x[idx+1]*y[idx+2] - x[idx+2]*y[idx+1]
        res[idx+1] = x[idx+2]*y[idx] - x[idx]*y[idx+2]
        res[idx+2] = x[idx]*y[idx+1] - x[idx+1]*y[idx]


@guvectorize([(float64[:], float64[:], float64[:])], '(n),(n)->(n)', target="cpu")
def cross_product_vectorized(x, y, res):
    shape = res.shape[0] // 3
    for i in range(shape):
        idx = i*3
        res[idx] = x[idx+1]*y[idx+2] - x[idx+2]*y[idx+1]
        res[idx+1] = x[idx+2]*y[idx] - x[idx]*y[idx+2]
        res[idx+2] = x[idx]*y[idx+1] - x[idx+1]*y[idx]


@guvectorize([(float64, float64, float64)], '(),()->()')
def fast_div(x, y, res):
    res = y/x