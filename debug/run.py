from mathematics.intersection_taichi import ray_triangle_hit
from mathematics.constants import EPS
import taichi as ti
from taichi_glsl.vector import vec3, normalize
from taichi_glsl.randgen import rand

ti.init(arch=ti.gpu, debug=True)
vertices = ti.Vector.field(n=3, dtype=ti.f32, shape=(3, 1))


@ti.func
def max_dim(v):
    dim = 0
    if v[0] > v[1]:
        if v[0] > v[2]:
            dim = 0
        else:
            dim = 2
    elif v[1] > v[2]:
        dim = 1
    else:
        dim = 2
    return dim


@ti.func
def permute(p, x, y, z):
    res = vec3(0.0, 0.0, 0.0)

    for i in ti.static(range(3)):
        if i == x:
            res[0] = p[i]
        elif i == y:
            res[1] = p[i]
        elif i == z:
            res[2] = p[i]

    return res


@ti.func
def ray_tri_hit2(p0, p1, p2, ro, rd, tMin, tMax):
    hit = 0
    t = 99.0
    p0t = p0-ro
    p1t = p1-ro
    p2t = p2-ro
    kz = max_dim(ti.abs(rd))
    kx = kz + 1
    if kx == 3:
        kx = 0
    ky = kx + 1
    if ky == 3:
        ky = 0
    d = permute(rd, kx, ky, kz)
    p0t = permute(p0t, kx, ky, kz)
    p1t = permute(p1t, kx, ky, kz)
    p2t = permute(p2t, kx, ky, kz)
    Sx = -d[0] / d[2]
    Sy = -d[1] / d[2]
    Sz = 1.0 / d[2]
    p0t[0] += Sx * p0t[2]
    p0t[1] += Sy * p0t[2]
    p1t[0] += Sx * p1t[2]
    p1t[1] += Sy * p1t[2]
    p2t[0] += Sx * p2t[2]
    p2t[1] += Sy * p2t[2]

    e0 = p1t[0] * p2t[1] - p1t[1] * p2t[0]
    e1 = p2t[0] * p0t[1] - p2t[1] * p0t[0]
    e2 = p0t[0] * p1t[1] - p0t[1] * p1t[0]

    if e0 == 0.0 or e1 == 0.0 or e2 == 0.0:
        p2txp1ty = ti.cast(p2t[0], ti.f64) * ti.cast(p1t[1], ti.f64)
        p2typ1tx = ti.cast(p2t[1], ti.f64) * ti.cast(p1t[0], ti.f64)
        e0 = ti.cast(p2typ1tx - p2txp1ty, ti.f32)
        p0txp2ty = ti.cast(p0t[0], ti.f64) * ti.cast(p2t[1], ti.f64)
        p0typ2tx = ti.cast(p0t[1], ti.f64) * ti.cast(p2t[0], ti.f64)
        e1 = ti.cast(p0txp2ty - p0typ2tx, ti.f32)
        p1txp0ty = ti.cast(p1t[0], ti.f64) * ti.cast(p0t[1], ti.f64)
        p1typ0tx = ti.cast(p1t[1], ti.f64) * ti.cast(p0t[0], ti.f64)
        e2 = ti.cast(p1typ0tx - p1txp0ty, ti.f32)
    
    if (e0 < 0 or e1 < 0 or e2 < 0) and (e0 > 0 or e1 > 0 or e2 > 0):
        hit = 0
    else:
        det = e0 + e1 + e2
        if det == 0:
            hit = 0
        else:
            p0t[2] *= Sz
            p1t[2] *= Sz
            p2t[2] *= Sz
            tScaled = e0 * p0t[2] + e1 * p1t[2] + e2 * p2t[2]
            if det < 0 and (tScaled >= 0 or tScaled < tMax * det):
                hit = 0
            elif det > 0 and (tScaled <= 0 or tScaled > tMax * det):
                hit = 0
            else:
                hit = 1
                invDet = 1.0 / det
                t = tScaled * invDet

                # barycentric coordinates (not needed for now)
                # b0 = e0 * invDet
                # b1 = e1 * invDet
                # b2 = e2 * invDet
    return hit, t


@ti.kernel
def main():
    vertices[0, 0] = vec3(rand(), rand(), rand())
    vertices[1, 0] = vec3(rand(), rand(), rand())
    vertices[2, 0] = vec3(rand(), rand(), rand())
    u = ti.sqrt(rand())
    v = rand()
    a = u * (1 - v)
    b = u * v
    rd = normalize(a*vertices[0, 0]+b*vertices[1, 0]+(1.0-a-b)*vertices[2, 0])
    ro = vec3(rand(), rand(), rand())
    hit, t = ray_triangle_hit(vertices[0, 0], vertices[1, 0], vertices[2, 0], ro, rd, 0.0, 9999.9)
    h2, t2 = ray_tri_hit2(vertices[0, 0], vertices[1, 0], vertices[2, 0], ro, rd, EPS, 9999.9)
    print(hit, t, h2, t2)
