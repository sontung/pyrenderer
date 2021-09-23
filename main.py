from io_utils.read_tungsten import read_file
from core.tracing import ray_casting
from tqdm import tqdm
import numpy as np
import random
import argparse
import cProfile, pstats, io


def trace_pixel(x, y, w, h, a_camera, a_scene):
    x = (x+random.random())/float(w)
    y = (y+random.random())/float(h)
    ray = a_camera.generate_ray(np.array([x, y]))
    return ray_casting(ray, a_scene)


def main():
    a_scene, a_camera = read_file("media/cornell-box/scene.json")
    x_dim, y_dim = a_camera.get_resolution()
    image = np.zeros((x_dim, y_dim, 3), dtype=np.float64)
    with tqdm(total=x_dim * y_dim, desc="rendering") as pbar:
        for i in range(x_dim):
            for j in range(y_dim):
                image[i, j] = trace_pixel(i, j, x_dim, y_dim, a_camera, a_scene)
                pbar.update(1)
    return image


def main_debug():
    a_scene, a_camera = read_file("media/cornell-box/scene.json")
    x_dim, y_dim = a_camera.get_resolution()
    image = np.zeros((x_dim, y_dim, 3), dtype=np.float64)
    for i in range(x_dim):
        for j in range(y_dim):
            image[i, j] = trace_pixel(i, j, x_dim, y_dim, a_camera, a_scene)
    return image


def main_profile():
    a_scene, a_camera = read_file("media/cornell-box/scene.json")
    x_dim, y_dim = a_camera.get_resolution()
    image = np.zeros((x_dim, y_dim, 3), dtype=np.float64)
    pr = cProfile.Profile()
    pr.enable()
    for i in range(20):
        for j in range(20):
            image[i, j] = trace_pixel(i, j, x_dim, y_dim, a_camera, a_scene)
    s = io.StringIO()
    sortby = 'time'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(100)
    print(s.getvalue())
    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', type=bool, default=False, help='Debug mode')
    parser.add_argument('-p', '--profile', type=bool, default=False, help='Profile mode')

    args = vars(parser.parse_args())
    DEBUG_MODE = args['debug']
    PROFILE_MODE = args['profile']
    if DEBUG_MODE:
        main_debug()
    elif PROFILE_MODE:
        main_profile()
    else:
        main()
