import numpy as np

Pi      = 3.14159265358979323846
InvPi   = 0.31830988618379067154
Inv2Pi  = 0.15915494309189533577
Inv4Pi  = 0.07957747154594766788
PiOver2 = 1.57079632679489661923
PiOver4 = 0.78539816339744830961
Sqrt2   = 1.41421356237309504880
EPS     = 1e-7  # anything smaller than this is considered zero
finfo = np.finfo(np.float64)
MAX_F = finfo.max

# rounding errors: https://www.pbr-book.org/3ed-2018/Shapes/Managing_Rounding_Error
MACHINE_EPS = finfo.eps * 0.5
GAMMA2_3 = (3 * MACHINE_EPS) / (1 - 3 * MACHINE_EPS)
