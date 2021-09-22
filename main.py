from io_utils.read_tungsten import read_file
from core.tracing import ray_casting
from tqdm import tqdm
import numpy as np
import random


def trace_pixel(x, y, w, h, a_camera, a_scene):
    x = (x+random.random())/float(w)
    y = (y+random.random())/float(h)
    ray = a_camera.generate_ray(np.array([x, y]))
    return ray_casting(ray, a_scene)


def main():
    a_scene, a_camera = read_file("media/cornell-box/scene.json")
    x_dim, y_dim = a_camera.get_resolution()
    image = np.zeros((x_dim, y_dim, 3), dtype=np.float32)
    with tqdm(total=x_dim * y_dim, desc="rendering") as pbar:
        for i in range(x_dim):
            for j in range(y_dim):
                image[i, j] = trace_pixel(i, j, x_dim, y_dim, a_camera, a_scene)
                pbar.update(1)
    return image


if __name__ == '__main__':
    main()
