import sys
from io_utils.read_tungsten import read_file
from core.ray import Ray
from core.tracing import ray_casting
from tqdm import tqdm
from skimage.io import imsave
import numpy as np
import random
import argparse
import cProfile, pstats, io
import open3d as o3d


def debug():
    sys.stdin = open("debug/nothit.txt", "r")
    lines = sys.stdin.readlines()
    a_scene, a_camera = read_file("media/cornell-box/scene.json")

    for line in lines:
        line = line[:-1]
        x, y, z, dx, dy, dz = map(float, line.split(" "))
        lines = []
        points = []
        colors = []

        ray = Ray(np.array([x, y, z]), np.array([dx, dy, dz]))
        ret = a_scene.hit(ray)
        ret2 = a_scene.hit_faster(ray)
        assert ret2["hit"] == ret["hit"]
        if ret["hit"]:
            assert abs(ret2["t"] - ret["t"]) < 0.001
        if not ret["hit"]:
            points.extend([ray.position, ray.position + 10 * ray.direction])
            lines.append([len(points) - 2, len(points) - 1])
            colors.append([1, 0, 0])
            # points.extend([ray.position, ray.position + 10 * -ray.direction])
            # lines.append([len(points) - 2, len(points) - 1])
            # colors.append([0, 1, 0])
        else:
            points.extend([ray.position, ray.position + ret["t"] * ray.direction])
            lines.append([len(points) - 2, len(points) - 1])
            colors.append([0, 1, 0])
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        # a_scene.visualize_o3d(line_set)

        # break
