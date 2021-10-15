import numpy as np
import taichi as ti
from mathematics.constants import EPS
from tqdm import tqdm

ti.init(arch=ti.gpu)


@ti.func
def fast_subtract(x, y, s):
    for i in range(x.shape[0]):
        s[i] = x[i]-y[i]


@ti.func
def cross_product(x, y, res):
    shape = res.shape[0] // 3
    for i in range(shape):
        idx = i*3
        res[idx] = x[idx+1]*y[idx+2] - x[idx+2]*y[idx+1]
        res[idx+1] = x[idx+2]*y[idx] - x[idx]*y[idx+2]
        res[idx+2] = x[idx]*y[idx+1] - x[idx+1]*y[idx]


@ti.func
def fast_dot3(x, ty, res):
    for i in range(0, x.shape[0]//3):
        res[i] = x[i*3]*ty[i*3]+x[i*3+1]*ty[i*3+1]+x[i*3+2]*ty[i*3+2]


@ti.func
def triangle_ray_intersection_numba(ind, ray_bound, a, e2r,
                                    sq, rdr, res_holder):
    if -EPS < a < EPS:
        res_holder[ind*2] = -1.0
    else:
        f = 1.0/a
        t = f*e2r
        if t > ray_bound[1] or t < EPS:
            res_holder[ind*2] = -1.0
        else:
            u = f*sq
            if u < 0.0:
                res_holder[ind*2] = -1.0
            else:
                v = f*rdr
                if v < 0.0 or u+v > 1.0:
                    res_holder[ind*2] = -1.0
                else:
                    res_holder[ind*2] = 1.0
                    res_holder[ind*2+1] = t
                    ray_bound[1] = t


@ti.kernel
def triangle_ray_intersection_grouping_numba(u1: ti.template(), u2: ti.template(), ray_bound: ti.template(),
                                             nb_triangles: ti.i32,
                                             s_array: ti.template(), q_array: ti.template(),
                                             r_array: ti.template(), p0_array: ti.template(),
                                             e1_array: ti.template(), e2_array: ti.template(),
                                             a_array: ti.template(), e2r_array: ti.template(),
                                             sq_array: ti.template(), rdr_array: ti.template(),
                                             res_holder: ti.template()):
    fast_subtract(u1, p0_array, s_array)
    cross_product(u2, e2_array, q_array)
    cross_product(s_array, e1_array, r_array)
    fast_dot3(e1_array, q_array, a_array)
    fast_dot3(e2_array, r_array, e2r_array)
    fast_dot3(s_array, q_array, sq_array)
    fast_dot3(u2, r_array, rdr_array)

    for i in range(nb_triangles):
        triangle_ray_intersection_numba(i, ray_bound, a_array[i], e2r_array[i], sq_array[i], rdr_array[i], res_holder)


u1 = np.random.rand(3)
u2 = np.random.rand(3)
bound = np.random.rand(2)
s_array = np.random.rand(12)
q_array = np.random.rand(12)
r_array = np.random.rand(12)
p0_array = np.random.rand(12)
e1_array = np.random.rand(12)
e2_array = np.random.rand(12)
a_array = np.random.rand(12)
e2r_array = np.random.rand(12)
sq_array = np.random.rand(12)
rdr_array = np.random.rand(12)
res_holder = np.random.rand(12)

u1_ti = ti.field(ti.f64, u1.shape)
u2_ti = ti.field(ti.f64, u2.shape)
ray_bounds = ti.field(ti.f64, bound.shape)
s_array_ti = ti.field(ti.f64, s_array.shape)
q_array_ti = ti.field(ti.f64, q_array.shape)
r_array_ti = ti.field(ti.f64, r_array.shape)
p0_array_ti = ti.field(ti.f64, p0_array.shape)
e1_array_ti = ti.field(ti.f64, e1_array.shape)
e2_array_ti = ti.field(ti.f64, e2_array.shape)
a_array_ti = ti.field(ti.f64, a_array.shape)
e2r_array_ti = ti.field(ti.f64, e2r_array.shape)
sq_array_ti = ti.field(ti.f64, sq_array.shape)
rdr_array_ti = ti.field(ti.f64, rdr_array.shape)
res_holder_ti = ti.field(ti.f64, res_holder.shape)

for _ in tqdm(range(10000)):
    u1 = np.random.rand(3)
    u2 = np.random.rand(3)
    bound = np.random.rand(2)
    s_array = np.random.rand(12)
    q_array = np.random.rand(12)
    r_array = np.random.rand(12)
    p0_array = np.random.rand(12)
    e1_array = np.random.rand(12)
    e2_array = np.random.rand(12)
    a_array = np.random.rand(12)
    e2r_array = np.random.rand(12)
    sq_array = np.random.rand(12)
    rdr_array = np.random.rand(12)
    res_holder = np.random.rand(12)

    u1_ti.from_numpy(u1)
    u2_ti.from_numpy(u2)
    ray_bounds.from_numpy(bound)
    s_array_ti.from_numpy(s_array)
    q_array_ti.from_numpy(q_array)
    r_array_ti.from_numpy(r_array)
    p0_array_ti.from_numpy(p0_array)
    e1_array_ti.from_numpy(e1_array)
    e2_array_ti.from_numpy(e2_array)
    a_array_ti.from_numpy(a_array)
    e2r_array_ti.from_numpy(e2r_array)
    sq_array_ti.from_numpy(sq_array)
    rdr_array_ti.from_numpy(rdr_array)
    res_holder_ti.from_numpy(res_holder)
    triangle_ray_intersection_grouping_numba(u1_ti, u2_ti, ray_bounds,
                                             3,
                                             s_array_ti, q_array_ti, r_array_ti, p0_array_ti,
                                             e1_array_ti, e2_array_ti, a_array_ti, e2r_array_ti,
                                             sq_array_ti, rdr_array_ti, res_holder_ti)
