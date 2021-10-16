import numpy as np
import taichi as ti
import time
from mathematics.intersection import triangle_ray_intersection_grouping_numba
from mathematics.constants import EPS
from tqdm import tqdm

ti.init(arch=ti.cpu)
inf = 1e6
eps = 1e-6


@ti.func
def fast_subtract(x, y, s):
    for i in ti.static(range(x.shape[0])):
        s[i] = x[i]-y[i]


@ti.func
def cross_product(x, y, res):
    for i in ti.static(range(res.shape[0] // 3)):
        idx = i*3
        res[idx] = x[idx+1]*y[idx+2] - x[idx+2]*y[idx+1]
        res[idx+1] = x[idx+2]*y[idx] - x[idx]*y[idx+2]
        res[idx+2] = x[idx]*y[idx+1] - x[idx+1]*y[idx]


@ti.func
def fast_dot3(x, ty, res):
    for i in ti.static(range(0, x.shape[0]//3)):
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

@ti.func
def ray_triangle_hit(v0, u, v, ro, rd):
    norm = u.cross(v)
    depth = ti.cast(inf * 2, ti.f64)
    s, t = ti.cast(0., ti.f64), ti.cast(0., ti.f64)
    hit = 0

    b = norm.dot(rd)
    if abs(b) >= eps:
        w0 = ro - v0
        a = -norm.dot(w0)
        r = a / b
        if r > 0:
            ip = ro + r * rd
            uu = u.dot(u)
            uv = u.dot(v)
            vv = v.dot(v)
            w = ip - v0
            wu = w.dot(u)
            wv = w.dot(v)
            D = uv * uv - uu * vv
            s = (uv * wv - vv * wu) / D
            t = (uv * wu - uu * wv) / D
            if 0 <= s <= 1:
                if 0 <= t and s + t <= 1:
                    depth = r
                    hit = 1
    return hit, depth


@ti.kernel
def triangle_ray_intersection_grouping_ti(u1: ti.template(), u2: ti.template(), p0_array: ti.template(),
                                          e1_array: ti.template(), e2_array: ti.template()):
    for _ in range(1):
        for i in range(12):
            ray_triangle_hit(p0_array[i, 0], e1_array[i, 0], e2_array[i, 0], u1[i, 0], u2[i, 0])

@profile
def main():
    u1 = np.random.rand(36)
    u2 = np.random.rand(36)
    bound = np.random.rand(2)
    s_array = np.random.rand(36)
    q_array = np.random.rand(36)
    r_array = np.random.rand(36)
    p0_array = np.random.rand(36)
    e1_array = np.random.rand(36)
    e2_array = np.random.rand(36)
    a_array = np.random.rand(12)
    e2r_array = np.random.rand(12)
    sq_array = np.random.rand(12)
    rdr_array = np.random.rand(12)
    res_holder = np.random.rand(24)

    u1_ti = ti.Vector.field(3, ti.f64, (12, 1))
    u2_ti = ti.Vector.field(3, ti.f64, (12, 1))
    p0_array_ti = ti.Vector.field(3, ti.f64, (12, 1))
    e1_array_ti = ti.Vector.field(3, ti.f64, (12, 1))
    e2_array_ti = ti.Vector.field(3, ti.f64, (12, 1))

    for i in range(12):
        u1_ti[i, 0] = u1[i * 3:i * 3 + 3]
        u2_ti[i, 0] = u2[i * 3:i * 3 + 3]
        p0_array_ti[i, 0] = p0_array[i * 3:i * 3 + 3]
        e1_array_ti[i, 0] = e1_array[i * 3:i * 3 + 3]
        e2_array_ti[i, 0] = e2_array[i * 3:i * 3 + 3]

    for _ in range(10000):
        # u1 = np.random.rand(36)
        # u2 = np.random.rand(36)
        # bound = np.random.rand(2)
        # s_array = np.random.rand(36)
        # q_array = np.random.rand(36)
        # r_array = np.random.rand(36)
        # p0_array = np.random.rand(36)
        # e1_array = np.random.rand(36)
        # e2_array = np.random.rand(36)
        # a_array = np.random.rand(12)
        # e2r_array = np.random.rand(12)
        # sq_array = np.random.rand(12)
        # rdr_array = np.random.rand(12)
        # res_holder = np.random.rand(24)

        triangle_ray_intersection_grouping_ti(u1_ti, u2_ti, p0_array_ti,
                                              e1_array_ti, e2_array_ti)

        triangle_ray_intersection_grouping_numba(u1, u2, bound,
                                                 12,
                                                 s_array, q_array, r_array, p0_array,
                                                 e1_array, e2_array, a_array, e2r_array,
                                                 sq_array, rdr_array, res_holder
                                                 )

main()
