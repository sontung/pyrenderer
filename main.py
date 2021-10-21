import sys
import time
from debug.ray_logger import RayLogger
from io_utils.read_tungsten import read_file
from core.tracing import ray_casting, path_tracing
from tqdm import tqdm
from skimage.io import imsave
import numpy as np
import random
import argparse
import cProfile, pstats, io
from joblib import Parallel, delayed
import open3d as o3d


def trace_pixel(x, y, w, h, a_camera, a_scene):
    total = np.zeros((3,), np.float32)
    for _ in range(SAMPLES):
        x = (x+random.random())/float(w)
        y = (y+random.random())/float(h)
        ray = a_camera.generate_ray(np.array([x, y]))
        e, r = path_tracing(ray, a_scene)
        color = e+r
        total = color + total
    return total/SAMPLES


def trace_pixel_par(x, y, w, h, a_camera, a_scene):
    total = np.zeros((3,), np.float32)
    for _ in range(SAMPLES):
        u = (x+random.random())/float(w)
        v = (y+random.random())/float(h)
        ray = a_camera.generate_ray(np.array([u, v]))
        e, r = path_tracing(ray, a_scene)
        color = e+r
        total = color + total
    return total/SAMPLES, x, y


def main():
    a_scene, a_camera = read_file("media/cornell-box/scene.json")
    x_dim, y_dim = a_camera.get_resolution()
    image = np.zeros((x_dim, y_dim, 3), dtype=np.float32)
    if SEQUENTIAL:
        with tqdm(total=x_dim * y_dim, desc="rendering") as pbar:
            for i in range(x_dim):
                for j in range(y_dim):
                    image[x_dim-1-j, i] = trace_pixel(i, j, x_dim, y_dim, a_camera, a_scene)
                    pbar.update(1)
    else:
        works = [(i, j) for i in range(x_dim) for j in range(y_dim)]
        results = Parallel(n_jobs=4)(delayed(trace_pixel_par)(i, j, x_dim, y_dim, a_camera, a_scene)
                                     for i, j in tqdm(works, desc="rendering"))
        for val, i, j in results:
            image[x_dim-1-j, i] = val

    image *= 255
    image = image.astype(np.uint8)
    imsave("test.png", image)
    return image


def main_debug():
    a_scene, a_camera = read_file("media/cornell-box/scene.json")

    x_dim, y_dim = a_camera.get_resolution()
    image = np.zeros((x_dim, y_dim, 3), dtype=np.float32)

    ray_logger = RayLogger()
    for i in range(0, x_dim, 10):
        for j in range(0, y_dim, 10):
            # if i != 404 or j != 951:
            #     continue
            for _ in range(6):
                x = (i + random.random()) / float(x_dim)
                y = (j + random.random()) / float(y_dim)
                ray = a_camera.generate_ray(np.array([x, y]))
                e, r = path_tracing(ray, a_scene, ray_logger)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(ray_logger.points)
    line_set.lines = o3d.utility.Vector2iVector(ray_logger.lines)
    line_set.colors = o3d.utility.Vector3dVector(ray_logger.colors)
    a_scene.visualize_o3d(line_set)
    return image


def main_profile():

    import taichi as ti
    ti.init(arch=ti.gpu)

    a_scene, a_camera = read_file("media/cornell-box/scene.json")
    x_dim, y_dim = a_camera.get_resolution()
    image = np.zeros((x_dim, y_dim, 3), dtype=np.float32)
    start = time.time()
    for i in range(0, x_dim, 50):
        for j in range(0, y_dim, 50):
            image[j, i] = trace_pixel(i, j, x_dim, y_dim, a_camera, a_scene)
    image *= 255
    image = image.astype(np.uint8)
    imsave("test.png", image)
    print(time.time() - start)

    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', type=bool, default=False, help='Debug mode')
    parser.add_argument('-p', '--profile', type=bool, default=False, help='Profile mode')
    parser.add_argument('-s', '--sequential', type=bool, default=False, help='Not to run in parallel')
    parser.add_argument('--samples', type=int, default=8, help='number of spp')

    args = vars(parser.parse_args())
    DEBUG_MODE = args['debug']
    PROFILE_MODE = args['profile']
    SEQUENTIAL = args['sequential']
    SAMPLES = args['samples']

    if DEBUG_MODE:
        main_debug()
    elif PROFILE_MODE:
        main_profile()
    else:
        main()
