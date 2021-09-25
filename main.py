from io_utils.read_tungsten import read_file
from core.tracing import ray_casting
from tqdm import tqdm
from skimage.io import imsave
import numpy as np
import random
import argparse
import cProfile, pstats, io
import open3d as o3d


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
                image[j, i] = trace_pixel(i, j, x_dim, y_dim, a_camera, a_scene)
                pbar.update(1)
    image *= 255
    image = image.astype(np.uint8)
    imsave("test.png", image)
    return image


def main_debug():
    a_scene, a_camera = read_file("media/cornell-box/scene.json")

    x_dim, y_dim = a_camera.get_resolution()
    image = np.zeros((x_dim, y_dim, 3), dtype=np.float64)
    lines = []
    points = []
    colors = []
    for i in range(0, x_dim, 100):
        for j in range(0, y_dim, 100):
            x = (i + random.random()) / float(x_dim)
            y = (j + random.random()) / float(y_dim)
            ray = a_camera.generate_ray(np.array([x, y]))
            ret = a_scene.hit(ray)
            if not ret["hit"]:
                points.extend([ray.position, ray.position+5*ray.direction])
                lines.append([len(points)-2, len(points)-1])
                colors.append([1, 0, 0])
            else:
                points.extend([ray.position, ray.position+ret["t"]*ray.direction])
                lines.append([len(points)-2, len(points)-1])
                colors.append([0, 1, 0])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    a_scene.visualize_o3d(line_set)
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
